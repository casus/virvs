"""
This script evaluates segmentation models (UNet and Pix2Pix) across multiple virus datasets 
using pixel-wise metrics such as SSIM, PSNR, and MSE. It computes segmentation quality 
for both foreground and background areas and reports various metrics.

Key Features:
- Evaluates two segmentation architectures: UNet and Pix2Pix
- Supports multiple virus datasets: HAdV, VACV, IAV, HSV, RV
- Computes pixel-wise metrics (SSIM, PSNR, MSE) for both foreground and background
- Handles single- and dual-channel image inputs
- Supports virus-specific image sizes and thresholds
- Computes mean metrics across multiple seeds for Pix2Pix evaluation

Input:
- Processed test images in TIFF format for each virus variant
- Precomputed ground truth masks in NumPy format
- Trained model weights for each model/virus pair

Output:
- Printed evaluation metrics per model and virus, including foreground and background metrics
"""

import argparse
from collections import defaultdict
import os
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity
from tqdm import tqdm
import csv
from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import round_to_1
from virvs.utils.metrics_utils import calculate_metrics
from datetime import datetime

def ssim_psnr(pred, label):
    """
    Calculates the Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR) 
    between the predicted and ground truth images.

    Args:
        pred (np.ndarray): The predicted image.
        label (np.ndarray): The ground truth image.

    Returns:
        tuple: SSIM and PSNR values as floats.
    """
    ssim = structural_similarity(
        np.squeeze(pred),
        np.squeeze(label),
        data_range=2,
    )
    mse = np.mean((pred - label) ** 2)  # Mean squared error
    psnr = 10 * np.log10((2**2) / mse)  # PSNR formula
    return ssim, psnr


def eval(dataloader, size, ch_in, dropout, weights, virus, threshold):
    """
    Evaluates a segmentation model on the test dataset using multiple metrics.
    It computes pixel-wise metrics like SSIM, PSNR, and MSE for both foreground 
    and background regions.

    Args:
        size (int): The size to which images are resized.
        ch_in (list): List of input channels (e.g., [0, 1] for dual-channel).
        dropout (bool): Whether dropout is applied to the model.
        model (str): The model type ('unet' or 'pix2pix').
        virus (str): The virus type for which to evaluate.
        threshold (float): Threshold used for mask generation.

    Returns:
        dict: Dictionary containing the mean values of various evaluation metrics.
    """
    generator = Generator(size, ch_in=ch_in, ch_out=1, apply_dropout=dropout)
    generator.load_weights(weights)  # Load the model weights

    test_metrics = defaultdict(list)

    # Iterate over the dataloader and compute metrics for each sample
    for sample in dataloader:
        x, y = sample  # Input image and ground truth mask
        output = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)  # Model prediction
        metrics = calculate_metrics(output, y)  # Calculate various metrics (e.g., IoU, F1)
        mask = y > threshold  # Create binary mask using threshold
        fg_ssim, fg_psnr = ssim_psnr(output[mask != 0], y[mask != 0])  # Compute SSIM and PSNR for foreground
        bg_ssim, bg_psnr = ssim_psnr(output[mask == 0], y[mask == 0])  # Compute SSIM and PSNR for background
        fg_mse = np.mean((output[mask != 0] - y[mask != 0]) ** 2)  # MSE for foreground
        bg_mse = np.mean((output[mask == 0] - y[mask == 0]) ** 2)  # MSE for background

        # Append metrics to test_metrics
        for k, v in metrics.items():
            test_metrics[k].append(v)
        test_metrics["fg_ssim"].append(fg_ssim)
        test_metrics["fg_psnr"].append(fg_psnr)
        test_metrics["fg_mse"].append(fg_mse)
        test_metrics["bg_ssim"].append(bg_ssim)
        test_metrics["bg_psnr"].append(bg_psnr)
        test_metrics["bg_mse"].append(bg_mse)

    # Calculate mean of all metrics
    for k, v in test_metrics.items():
        test_metrics[k] = np.mean(np.array(v))

    return test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['unet', 'pix2pix'])
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--virus')

    args = parser.parse_args()
    virus = args.virus

    # Set size and threshold based on virus type
    if "vacv" == virus:
        size = 5888
    else:
        size = 2048

    if "hadv" in virus:
        threshold = -0.9
    else:
        threshold = -0.8

    # Set channels based on virus type (single or dual-channel)
    if "2ch" in virus:
        ch_in = [0, 1]
    else:
        ch_in = [0]

    # Create dataloader for test dataset
    dataloader = NpyDataloader(
        path=args.dataset,
        im_size=size,
        random_jitter=False,
        crop_type="center",
        ch_in=ch_in,
    )

    output_dir = os.path.join("/bigdata/casus/MLID/maria/outputs/evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"metrics_{args.virus}_{timestamp}.csv")
                                
    # Evaluate for both models (UNet and Pix2Pix)
    dropout = args.model != "unet"  # Apply dropout only for Pix2Pix
    print(args.model, ", ", args.virus)

    if args.model == 'unet':
            tf.keras.utils.set_random_seed(42)  # Set random seed for reproducibility
            test_metrics = eval(dataloader, size, ch_in, dropout, args.weights, virus, threshold)

            # Print evaluation metrics
            for key, value in test_metrics.items():
                print(f"{key}: {round(value, 3)}")
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Metric', 'Value'])
                for key, value in test_metrics.items():
                    writer.writerow([key, round(value, 5)])

    else:
        seeds = [42, 43, 44]  # Seeds for averaging Pix2Pix metrics
        all_seeds_metrics = defaultdict(list)

        # Evaluate Pix2Pix model across multiple seeds
        for seed in seeds:
            tf.keras.utils.set_random_seed(seed)
            test_metrics = eval(size, ch_in, dropout, args.weights, virus, threshold)
            for k, v in test_metrics.items():
                all_seeds_metrics[k].append(np.mean(v))

        # Calculate and print mean and std for each metric
        results = {}
        for k, v in all_seeds_metrics.items():
            mean_metric = np.mean(v)
            std_metric = np.std(v)
            results[k] = {"mean": mean_metric, "std": std_metric}

        for key, value in results.items():
            print(f"{key}: mean = {round(value['mean'], 3)}, std = {round_to_1(value['std']):.1e}")

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in test_metrics.items():
                writer.writerow([key, round(value, 5)])

main()