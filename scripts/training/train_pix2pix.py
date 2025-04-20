"""
This script trains a Pix2Pix model for image-to-image translation tasks, specifically for 
segmentation of viral images. The model uses adversarial training with a generator and 
discriminator, optimizing both the adversarial loss and pixel-wise L1 loss. The script also 
supports logging training and validation metrics (including losses and evaluation metrics) to 
Neptune for tracking purposes.

Key Features:
- Trains a Pix2Pix model with a generator and discriminator for image segmentation tasks
- Supports logging of training metrics and model weights to Neptune for tracking
- Evaluates the model with simple pixelwise metrics 

Input:
- Training and validation image datasets with corresponding ground truth images (masks)
- A configuration file specifying model architecture, training settings, and Neptune integration

Output:
- Trained Pix2Pix model with weights saved after training completion
- Logged training and validation metrics to Neptune
"""

import argparse
import uuid
from collections import defaultdict

import neptune as neptune
import numpy as np
import tensorflow as tf
from skimage.util import montage

from virvs.architectures.pix2pix import Discriminator, Generator
from virvs.configs.utils import (
    create_data_config,
    create_eval_config,
    create_neptune_config,
    create_training_config,
    load_config_from_yaml,
)
from virvs.utils.inference_utils import log_metrics, save_output_montage, save_weighs
from virvs.utils.metrics_utils import calculate_metrics
from virvs.utils.training_utils import prepare_dataset

# Set a random seed for reproducibility
tf.keras.utils.set_random_seed(42)

def generator_loss(discriminator_generated_output, generator_output, ground_truth):
    """
    Compute the generator loss, including adversarial loss and L1 loss.
    
    Args:
        discriminator_generated_output (tensor): The discriminator's output for generated images.
        generator_output (tensor): The output from the generator.
        ground_truth (tensor): The ground truth (target) images.

    Returns:
        tuple: Total generator loss, adversarial loss, and L1 loss.
    """
    # Adversarial loss (binary cross entropy)
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_generated_output), discriminator_generated_output
    )
    
    # L1 loss (mean absolute error between the ground truth and generated images)
    l1_loss = tf.reduce_mean(tf.abs(ground_truth - generator_output))
    
    # Total loss is the sum of the adversarial loss and L1 loss, with a weight on L1 loss
    total_loss = gen_loss + 100 * l1_loss
    return total_loss, gen_loss, l1_loss


def discriminator_loss(
    discriminator_real_output, discriminator_generated_output, disc_weight
):
    """
    Compute the discriminator loss for distinguishing real and fake images.
    
    Args:
        discriminator_real_output (tensor): The discriminator's output for real images.
        discriminator_generated_output (tensor): The discriminator's output for generated images.
        disc_weight (float): The weight applied to the discriminator's loss.

    Returns:
        tensor: The total discriminator loss, scaled by disc_weight.
    """
    # Loss for real images (real images should be classified as real)
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_real_output), discriminator_real_output
    )
    
    # Loss for generated (fake) images (generated images should be classified as fake)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(discriminator_generated_output), discriminator_generated_output
    )
    
    # Total discriminator loss is the sum of the real and generated losses
    total_loss = real_loss + generated_loss
    return total_loss * disc_weight


@tf.function
def train_step(
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    input_image,
    target,
    disc_weight,
):
    """
    Perform one step of training, updating both the generator and the discriminator.
    
    Args:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
        generator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        discriminator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
        input_image (tensor): The input image to the generator.
        target (tensor): The target (ground truth) image.
        disc_weight (float): The weight applied to the discriminator's loss.
    
    Returns:
        tuple: Generator output, total generator loss, generator loss, L1 loss, and discriminator loss.
    """
    # Record operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate the output from the generator
        gen_output = generator(input_image, training=True)

        # Get the discriminator's output for real and generated images
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Calculate the generator's losses
        gen_total_loss, gen_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        
        # Calculate the discriminator's loss
        disc_total_loss = discriminator_loss(
            disc_real_output, disc_generated_output, disc_weight
        )

    # Compute gradients for both generator and discriminator
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_total_loss, discriminator.trainable_variables
    )

    # Apply gradients to update the weights of the models
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss


def main():
    """
    Main function that handles model training, evaluation, and logging.
    
    - Loads configuration files
    - Prepares datasets
    - Initializes the model (generator, discriminator)
    - Trains the model
    - Logs metrics to Neptune
    - Saves the model weights and outputs
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )
    parser.add_argument("--neptune-token", help="API token for Neptune")

    args = parser.parse_args()

    # Print available CPUs and GPUs for training
    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Load configuration files for data, training, evaluation, and Neptune
    config = load_config_from_yaml(args.config_path)
    data_config = create_data_config(config)
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    # Assert that data paths are provided in the configuration
    assert data_config.train_data_path is not None
    assert data_config.val_data_path is not None

    # Set input and output channels
    channels_in = data_config.ch_in
    channels_out = 1

    # Prepare training and validation datasets
    dataset = prepare_dataset(
        path=data_config.train_data_path,
        im_size=data_config.im_size,
        random_jitter=True,
        ch_in=channels_in,
        ch_out=channels_out,
    )
    val_size = 2048

    val_dataset = prepare_dataset(
        path=data_config.val_data_path,
        im_size=val_size,
        ch_in=channels_in,
        ch_out=channels_out,
    )

    batch_size = data_config.batch_size
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(batch_size)
    val_dataset = val_dataset.batch(1)

    # Initialize the generator and discriminator models
    generator = Generator(
        data_config.im_size,
        ch_in=channels_in,
        ch_out=channels_out,
        apply_batchnorm=True,
    )
    discriminator = Discriminator(
        data_config.im_size, ch_in=channels_in, ch_out=channels_out
    )
    generator.summary()

    # Set up optimizers for the generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    output_path = eval_config.output_path

    # Initialize Neptune for logging (if a token is provided)
    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    print("Starting training...")

    # Initialize training loop parameters
    step = 0
    log_freq = eval_config.log_freq
    val_freq = eval_config.val_freq
    max_steps = training_config.max_steps
    run_id = str(uuid.uuid4())  # Unique ID for this run
    cumulative_loss = np.zeros(4)
    train_metrics = defaultdict(float)

    # Main training loop
    while True:
        for batch in dataset:
            batch_x, batch_y = batch

            # Perform a training step
            output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss = train_step(
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                batch_x,
                batch_y,
                training_config.pix2pix_disc_weight,
            )

            # Update cumulative loss and calculate metrics
            cumulative_loss += np.array(
                [gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss]
            )
            metrics = calculate_metrics(output, batch_y.numpy())
            for k, v in metrics.items():
                train_metrics[k] += v

            # Log metrics periodically
            if step % log_freq == 0:
                if run is not None:
                    run[f"train_loss_total"].log(
                        (cumulative_loss[0] + cumulative_loss[3]) / log_freq
                    )
                    run[f"train_loss_gen_total"].log((cumulative_loss[0]) / log_freq)
                    run[f"train_loss_gen"].log((cumulative_loss[1]) / log_freq)
                    run[f"train_loss_l1"].log((cumulative_loss[2]) / log_freq)
                    run[f"train_loss_disc_total"].log((cumulative_loss[3]) / log_freq)
                    cumulative_loss = np.zeros(4)

                    # Log detailed metrics
                    for k, v in metrics.items():
                        train_metrics[k] /= log_freq
                    log_metrics(run, train_metrics, prefix="train")
                    train_metrics = defaultdict(float)

            # Validate the model periodically
            if step % val_freq == 0:
                val_metrics = defaultdict(float)

                # Copy weights from the generator for validation to a generator of full size
                weights = generator.get_weights()
                generator_val = Generator(
                    2048,
                    ch_in=channels_in,
                    ch_out=channels_out,
                    apply_batchnorm=True,
                )
                generator_val.set_weights(weights)

                for n, batch in enumerate(val_dataset):
                    batch_x, batch_y = batch
                    # Use training=True to use batch normalization & dropout
                    output = generator_val(batch_x, training=True)

                    metrics = calculate_metrics(output, batch_y.numpy())
                    for k, v in metrics.items():
                        val_metrics[k] += v

                # Average validation metrics over all batches
                for k, v in val_metrics.items():
                    val_metrics[k] = val_metrics[k] / (n + 1)

                log_metrics(run, val_metrics, prefix="val")
            # Save weights and stop training if max steps reached
            if step == max_steps:
                if run is not None:
                    save_weighs(
                        run=run,
                        model=generator,
                        step=step,
                        output_path=output_path,
                        run_id=run_id,
                    )
                    run.stop()
                exit(0)
            step += 1


if __name__ == "__main__":
    main()
