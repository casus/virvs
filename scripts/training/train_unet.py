"""
This script trains a UNet model for image-to-image translation tasks, 
focusing on the segmentation of viral images.
Training and validation metrics are logged to Neptune, and model weights are saved periodically.

Key Features:
- Trains a UNet model for image segmentation tasks
- Supports logging of training metrics and model weights to Neptune for tracking

Input:
- Training and validation image datasets with corresponding ground truth images (masks)
- A configuration file specifying model architecture, training settings, and Neptune integration

Output:
- Trained UNet model with weights saved after training completion
- Logged training and validation metrics (loss, accuracy, etc.) to Neptune
"""

import argparse
from datetime import datetime
import uuid
from collections import defaultdict

import neptune as neptune
import numpy as np
import tensorflow as tf
from skimage.util import montage
import os 

from virvs.architectures.pix2pix import Generator
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

# Setting a fixed random seed for reproducibility
tf.keras.utils.set_random_seed(42)


@tf.function
def train_step(generator, generator_optimizer, input_image, target):
    """
    Performs a single training step for the generator model.
    
    Args:
        generator: The generator model to train.
        generator_optimizer: The optimizer to apply gradients.
        input_image: Input image batch for the generator.
        target: The ground truth output corresponding to input_image.

    Returns:
        Tuple of (generator output, computed loss)
    """
    with tf.GradientTape() as gen_tape:
        # Forward pass: generate an output for the given input
        gen_output = generator(input_image, training=True)
        # Compute loss as mean absolute difference between output and target
        loss = tf.reduce_mean(tf.abs(gen_output - target))

    # Compute gradients with respect to loss
    generator_gradients = gen_tape.gradient(loss, generator.trainable_variables)
    # Apply gradients to update the model
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    
    return gen_output, loss


def main():
    """
    Main function that sets up training, including loading configurations, 
    preparing datasets, initializing models, and starting the training loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )
    parser.add_argument("--neptune-token", help="API token for Neptune")

    # Parse command-line arguments
    args = parser.parse_args()

    # Print available hardware (CPUs, GPUs)
    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Load configuration from YAML file
    RANDOM_SEED = os.environ["RANDOM_SEED"]

    config = load_config_from_yaml(args.config_path)
    data_config = create_data_config(config)
    data_config.train_data_path = data_config.train_data_path.replace("processed", f"processed_{RANDOM_SEED}")
    data_config.val_data_path = data_config.val_data_path.replace("processed", f"processed_{RANDOM_SEED}")
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    print("Getting data...")

    # Ensure paths for training and validation data are provided
    assert data_config.train_data_path is not None
    assert data_config.val_data_path is not None

    # Set input and output channels (grayscale image output)
    channels_in = data_config.ch_in
    channels_out = 1

    # Prepare datasets for training and validation
    dataset = prepare_dataset(
        path=data_config.train_data_path,
        im_size=data_config.im_size,
        random_jitter=True,
        ch_in=channels_in,
        ch_out=channels_out,
    )
    val_dataset = prepare_dataset(
        path=data_config.val_data_path,
        im_size=2048,
        ch_in=channels_in,
        ch_out=channels_out,
    )

    # Set batch size
    batch_size = data_config.batch_size

    # Shuffle and batch datasets
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(batch_size)
    val_dataset = val_dataset.batch(1)

    # Initialize the generator model
    generator = Generator(
        256, ch_in=channels_in, ch_out=channels_out, apply_batchnorm=True
    )
    generator.summary()

    # Initialize optimizer (Adam)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Output path for saving model weights
    output_path = eval_config.output_path

    # Initialize Neptune run if Neptune API token is provided
    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    print("Starting training...")

    step = 0
    log_freq = eval_config.log_freq  # Frequency for logging training metrics
    val_freq = eval_config.val_freq  # Frequency for validating the model
    max_steps = training_config.max_steps  # Maximum number of training steps

    run_id = (
        RANDOM_SEED +  # Unique identifier prefix
        "_" +  # Separator
        datetime.now().strftime("%Y%m%d_%H%M%S")  # Current datetime in compact format
    )
    cumulative_loss = 0.0
    train_metrics = defaultdict(float)  # Dictionary to store accumulated metrics during training

    # Main training loop
    while True:
        for batch in dataset:
            batch_x, batch_y = batch  # Input and target images
            # Perform a training step
            output, gen_total_loss = train_step(
                generator,
                generator_optimizer,
                batch_x,
                batch_y,
            )
            cumulative_loss += gen_total_loss
            # Calculate additional metrics for tracking progress
            metrics = calculate_metrics(output, batch_y.numpy())
            for k, v in metrics.items():
                train_metrics[k] += v

            # Log training metrics at specified frequency
            if step % log_freq == 0:
                if run is not None:
                    run[f"train_loss_total"].log(cumulative_loss / log_freq)
                    cumulative_loss = 0.0

                    for k, v in metrics.items():
                        train_metrics[k] /= log_freq
                    log_metrics(run, train_metrics, prefix="train")
                    train_metrics = defaultdict(float)

            # Perform validation at specified frequency
            if step % val_freq == 0:
                val_metrics = defaultdict(float)
                # Copy weights from the generator for validation to a generator of full size
                weights = generator.get_weights()
                generator_val = Generator(
                    2048,
                    ch_in=channels_in,
                    ch_out=channels_out,
                    apply_batchnorm=True,
                    apply_dropout=False,
                )
                generator_val.set_weights(weights)
                # Validate over the validation dataset
                for n, batch in enumerate(val_dataset):
                    batch_x, batch_y = batch
                    output = generator_val(batch_x, training=True)
                    metrics = calculate_metrics(output, batch_y.numpy())
                    for k, v in metrics.items():
                        val_metrics[k] += v

                # Average validation metrics over all validation steps
                for k, v in val_metrics.items():
                    val_metrics[k] = val_metrics[k] / (n + 1)

                # Log validation metrics
                log_metrics(run, val_metrics, prefix="val")

            # Stop training once the maximum number of steps is reached
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