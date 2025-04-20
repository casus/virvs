"""
This script trains a Pix2Pix model for image-to-image translation tasks, specifically for 
segmentation of viral images. It performs adversarial training using a generator and a 
discriminator network, optimizing pixel-wise and adversarial losses. The model is evaluated 
on multiple metrics, including accuracy, F1-score, IoU, precision, and recall.
The calcuation of these metrics requires more computation - if you don't need them, use train_pix2pix.py.

Key Features:
- Trains a Pix2Pix model with a generator and discriminator for image segmentation
- Supports logging of training metrics and model weights to Neptune for tracking
- Evaluates model performance using segmentation metrics (IoU, F1-score, accuracy, etc.)

Input:
- Training and validation image datasets with corresponding ground truth masks
- Configurations for model architecture, training parameters, and Neptune integration

Output:
- Trained Pix2Pix model with weights saved at the end of the training
- Logged training and validation metrics  to Neptune
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
from virvs.utils.evaluation_utils import (
    calculate_acc,
    calculate_acc_only_cells,
    calculate_f1,
    calculate_iou,
    calculate_prec,
    calculate_rec,
    get_masks_to_show,
    get_mean_per_mask,
)
from virvs.utils.inference_utils import log_metrics, save_output_montage, save_weighs
from virvs.utils.metrics_utils import calculate_metrics
from virvs.utils.training_utils import prepare_dataset

# Set a random seed for reproducibility
tf.keras.utils.set_random_seed(42)

# Loss function for the generator (adversarial + L1 loss)
def generator_loss(discriminator_generated_output, generator_output, ground_truth):
    """
    Compute the generator loss which includes adversarial loss and L1 loss.
    
    Args:
        discriminator_generated_output (tensor): The output from the discriminator for generated images.
        generator_output (tensor): The output from the generator.
        ground_truth (tensor): The ground truth (target) images.

    Returns:
        tuple: Total generator loss, adversarial loss, and L1 loss.
    """
    # Adversarial loss: Encourages the generator to generate realistic images
    gen_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_generated_output), discriminator_generated_output
    )

    # L1 loss: Pixel-wise absolute difference between generated and ground truth images
    l1_loss = tf.reduce_mean(tf.abs(ground_truth - generator_output))

    # The total loss is a combination of adversarial loss and L1 loss
    total_loss = gen_loss + 100 * l1_loss
    return total_loss, gen_loss, l1_loss

# Loss function for the discriminator (real vs fake)
def discriminator_loss(
    discriminator_real_output, discriminator_generated_output, disc_weight
):
    """
    Compute the discriminator loss to differentiate between real and fake images.
    
    Args:
        discriminator_real_output (tensor): The discriminator's output for real images.
        discriminator_generated_output (tensor): The discriminator's output for generated images.
        disc_weight (float): Weight applied to the discriminator's loss.

    Returns:
        tensor: The total discriminator loss, scaled by disc_weight.
    """
    # Loss for real images (encourages discriminator to classify real images as real)
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(discriminator_real_output), discriminator_real_output
    )

    # Loss for generated images (encourages discriminator to classify fake images as fake)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(discriminator_generated_output), discriminator_generated_output
    )

    # Total loss is the sum of real and generated losses
    total_loss = real_loss + generated_loss
    return total_loss * disc_weight

# Function for a single training step (updates generator and discriminator)
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
    Perform one step of training (updates both generator and discriminator).
    
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
    # Record the operations for automatic differentiation (backpropagation)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate the output from the generator
        gen_output = generator(input_image, training=True)

        # Get discriminator's output for real and generated images
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        # Compute losses
        gen_total_loss, gen_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
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

    # Apply gradients to update the weights
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    # Return various loss values for later logging and analysis
    return gen_output, gen_total_loss, gen_loss, gen_l1_loss, disc_total_loss

# Prepares masks for highlighted regions based on GFP signal
def prepare_highlighted_masks(masks, gfp):
    """
    Prepare masks for highlighted regions where GFP signal is strong.
    
    Args:
        masks (np.ndarray): Ground truth masks.
        gfp (np.ndarray): The GFP signal (predicted or ground truth).
    
    Returns:
        np.ndarray: New masks indicating highlighted regions.
    """
    # Consider GFP signal above a threshold to identify highlighted regions
    pred_weights = (np.squeeze(gfp).flatten() > -0.9).astype(np.float32)
    
    # Compute mean values per mask to highlight significant regions
    mean_per_mask = get_mean_per_mask(masks, pred_weights)
    
    # Determine which masks to display based on their mean values
    masks_to_show = get_masks_to_show(mean_per_mask, 0.5)
    
    # Create a mask indicating the highlighted regions
    new_mask = np.isin(masks, masks_to_show)
    return new_mask

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

    # Load configuration files
    config = load_config_from_yaml(args.config_path)
    data_config = create_data_config(config)
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    # Ensure that data paths are provided in the configuration
    assert data_config.train_data_path is not None
    assert data_config.val_data_path is not None

    # Prepare datasets for training and validation
    dataset = prepare_dataset(
        path=data_config.train_data_path,
        im_size=data_config.im_size,
        random_jitter=True,
        ch_in=data_config.ch_in,
        ch_out=1,
    )
    val_dataset = prepare_dataset(
        path=data_config.val_data_path,
        im_size=2048,
        ch_in=data_config.ch_in,
        ch_out=1,
    )

    batch_size = data_config.batch_size
    dataset = dataset.shuffle(5000)
    val_dataset = val_dataset.batch(1)
    masks = np.load("/bigdata/casus/MLID/maria/VIRVS_data/masks/masks_nuc_hadv_val.npy")

    # Batch datasets
    dataset = dataset.batch(batch_size)

    # Initialize models (generator and discriminator)
    generator = Generator(
        data_config.im_size,
        ch_in=data_config.ch_in,
        ch_out=1,
        apply_batchnorm=True,
    )
    discriminator = Discriminator(
        data_config.im_size, ch_in=data_config.ch_in, ch_out=1
    )

    # Print generator summary
    generator.summary()

    # Initialize optimizers for generator and discriminator
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Output path for saving models
    output_path = eval_config.output_path

    # Initialize Neptune run (if token is provided)
    run = None
    if args.neptune_token is not None and neptune_config is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    print("Starting training...")

    # Training parameters
    step = 0
    log_freq = eval_config.log_freq
    val_freq = 2000
    max_steps = training_config.max_steps
    run_id = str(uuid.uuid4())  # Unique run ID for model saving

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

            # Update cumulative losses and metrics
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
                    for k, v in metrics.items():
                        train_metrics[k] /= log_freq
                    log_metrics(run, train_metrics, prefix="train")
                    train_metrics = defaultdict(float)

            # Validate periodically
            if step % val_freq == 0:
                # Create a new generator for full size of data for validation and set its weights 
                weights = generator.get_weights()
                generator_val = Generator(
                    2048,
                    ch_in=data_config.ch_in,
                    ch_out=1,
                    apply_batchnorm=True,
                )
                generator_val.set_weights(weights)

                ious, accs, precs, recs, f1s, acc_only_cells = [], [], [], [], [], []
                for n, batch in enumerate(val_dataset):
                    batch_x, batch_y = batch
                    # Use training=True to use batch normalization & dropout
                    output = generator_val(batch_x, training=True)
                    for i in range(len(batch_x)):
                        mask = masks[n + i]
                        gt_masks = prepare_highlighted_masks(
                            mask.flatten(),
                            batch_y[i].numpy().flatten(),
                        )
                        pred_masks = prepare_highlighted_masks(
                            mask.flatten(),
                            output[i].numpy().flatten(),
                        )
                        if np.sum(gt_masks) == 0:
                            continue
                        iou = calculate_iou(pred_masks, gt_masks)
                        ious.append(iou)
                        f1 = calculate_f1(gt_masks, pred_masks, np.sum(mask == 0))
                        f1s.append(f1)
                        acc = calculate_acc(pred_masks, gt_masks)
                        accs.append(acc)
                        rec = calculate_rec(pred_masks, gt_masks)
                        recs.append(rec)
                        prec = calculate_prec(pred_masks, gt_masks)
                        precs.append(prec)
                        acc_only_cells.append(calculate_acc_only_cells(pred_masks, gt_masks))

                # Compute mean metrics
                metrics = {
                    "iou": np.mean(ious),
                    "f1": np.mean(f1s),
                    "accuracy": np.mean(accs),
                    "precision": np.mean(precs),
                    "recall": np.mean(recs),
                    "acc_only_cells": np.mean(acc_only_cells),
                }

                # Log validation metrics
                if run is not None:
                    log_metrics(run, metrics, prefix="val")

                # Save output montage and model weights
                save_output_montage(output, step, run_id, output_path)
                save_weighs(generator, step, run_id, output_path)

            # Stop training if maximum steps are reached
            if step >= max_steps:
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
