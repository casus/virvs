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

from collections import defaultdict

import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity
from tqdm import tqdm

from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import round_to_1
from virvs.utils.metrics_utils import calculate_metrics


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


BASE_PATH = "/home/wyrzyk93/VIRVS/outputs/weights/"
DATASETS = {
    "hadv_2ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "hadv_1ch": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "vacv": "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed/test",
    "iav": "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed/test",
    "hsv": "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed/test",
    "rv": "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed/test",
}

WEIGHTS = {
    "pix2pix": {
        "hadv_2ch": "model_200000_38527895-6967-4f7b-a5df-eb6564b47ad9.h5",
        "hadv_1ch": "model_200000_63065c3c-b6bd-4a38-a5b9-9d9ae5083129.h5",
        "vacv": "model_200000_8b365816-a14c-4279-885f-f61db645e786.h5",
        "iav": "model_200000_fbc9c491-14a6-4ed4-a02d-daefd99fd60e.h5",
        "hsv": "model_200000_afb8f712-ae29-404e-bcd9-a06786e5952b.h5",
        "rv": "model_200000_964056e1-a1e3-4c4f-9e9e-21112f54453a.h5",
    },
    "unet": {
        "hadv_2ch": "model_200000_633f9e8a-0c3b-4678-9a67-ec351fdf09de.h5",
        "hadv_1ch": "model_200000_b301ae59-b3a9-4faf-9ae1-e3b805712147.h5",
        "vacv": "model_200000_d6b695f9-fa90-4ce2-ba99-958078cabd52.h5",
        "iav": "model_200000_868408c5-7b76-412b-9a8d-faf4812ed96f.h5",
        "hsv": "model_200000_1d7dded0-481b-4a29-b3d5-16899f219b6e.h5",
        "rv": "model_200000_d88881a7-e0c3-455f-bb5d-521cfbb98329.h5",
    },
}


def eval(size, ch_in, dropout, model, virus, threshold):
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
    generator.load_weights(f"{BASE_PATH}/{WEIGHTS[model][virus]}")  # Load the model weights

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


# Iterate through datasets and models to evaluate
for virus in DATASETS.keys():

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
        path=DATASETS[virus],
        im_size=size,
        random_jitter=False,
        crop_type="center",
        ch_in=ch_in,
    )

    # Evaluate for both models (UNet and Pix2Pix)
    for model in ["unet", "pix2pix"]:
        dropout = model != "unet"  # Apply dropout only for Pix2Pix
        print(model, ", ", virus)

        if model == "unet":
            tf.keras.utils.set_random_seed(42)  # Set random seed for reproducibility
            test_metrics = eval(size, ch_in, dropout, model, virus, threshold)

            # Print evaluation metrics
            for key, value in test_metrics.items():
                print(f"{key}: {round(value, 3)}")

        else:
            seeds = [42, 43, 44]  # Seeds for averaging Pix2Pix metrics
            all_seeds_metrics = defaultdict(list)

            # Evaluate Pix2Pix model across multiple seeds
            for seed in seeds:
                tf.keras.utils.set_random_seed(seed)
                test_metrics = eval(size, ch_in, dropout, model, virus, threshold)
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
