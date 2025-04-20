"""
This script processes virus-infected cell microscopy images to segment nuclei using Cellpose.
It processes test datasets for four different viruses (HAdV, IAV, HSV, RV), saves the predicted
nuclear masks as numpy arrays for downstream analysis.

Key Features:
- Uses Cellpose's pretrained 'cyto3' model for nucleus segmentation
- Handles multiple virus datasets with customizable diameter parameters
- Processes TIFF images and saves masks in numpy format
- Includes progress tracking with tqdm

Input: TIFF images organized in virus-specific directories
Output: Numpy arrays containing nuclear masks for each virus dataset
"""

import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tif
from cellpose import models, utils
from tqdm import tqdm

# Dictionary containing paths to test datasets for different viruses
DATASETS = {
    "hadv": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "iav": "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed/test",
    "hsv": "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed/test",
    "rv": "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed/test",
}

# Dictionary containing diameter parameters for Cellpose segmentation
# None means Cellpose will estimate diameter automatically
DIAMETERS = {
    "hadv": None,  # Automatic diameter estimation for HAdV
    "hsv": 7,      # Fixed diameter of 7 pixels for HSV
    "iav": 7,      # Fixed diameter of 7 pixels for IAV
    "rv": 7,       # Fixed diameter of 7 pixels for RV
}

# Set random seed for reproducibility
tf.random.set_seed(42)

# Initialize Cellpose model with 'cyto3' pretrained model and GPU acceleration
model = models.CellposeModel(model_type="cyto3", gpu=True)

# Process each virus dataset
for virus in DATASETS.keys():
    masks = []  # List to store masks for all images in current virus dataset
    
    # Get path to current virus dataset
    path = f"{DATASETS[virus]}"
    
    # Get list of all image files in the 'x' subdirectory
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    
    # Process each image with progress bar
    for f in tqdm(filenames, desc=f"Processing {virus}"):
        # Read image
        x = tif.imread(join(path, "x", f))[..., 0]
        
        # Run Cellpose prediction
        # channels=[0,0] means we're using nuclei channel
        masks_pred, flows, _ = model.eval(x, channels=[0, 0], diameter=DIAMETERS[virus])
        masks.append(masks_pred)
    
    # Create output directory if it doesn't exist
    os.makedirs(join("/bigdata/casus/MLID/maria/VIRVS_data", "masks"), exist_ok=True)
    
    # Save all masks for current virus as numpy array
    np.save(
        join(
            "/bigdata/casus/MLID/maria/VIRVS_data/masks",
            f"masks_nuc_{virus}_test.npy",
        ),
        np.array(masks),
    )