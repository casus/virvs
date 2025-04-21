"""
This script processes infected cell microscopy images to segment
whole cells using Cellpose. It processes train, validation, and test splits of the dataset,
saving the predicted cell masks as numpy arrays for downstream analysis.

Key Features:
- Uses Cellpose's pretrained 'cyto3' model for cell segmentation
- Processes three dataset splits (train/val/test)
- Handles 3-channel TIFF images with specific channel arrangement
- Saves masks in numpy format with organized filenames
- Includes progress tracking with tqdm

Input: 3-channel TIFF images organized in split-specific directories (train/val/test)
Output: Numpy arrays containing cell masks for each dataset split
"""

import os
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tifffile as tif
from cellpose import models, utils
from tqdm import tqdm

# Path to the HAdV dataset containing train/val/test splits
DATASET = "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed"

# Initialize Cellpose model with 'cyto3' pretrained model and GPU acceleration
model = models.CellposeModel(model_type="cyto3", gpu=True)

# Set random seed for reproducibility
tf.random.set_seed(42)

# Process each dataset split (train, validation, test)
for split in ["train", "val", "test"]:
    masks = []  # List to store masks for all images in current split
    
    # Get path to current split directory
    path = f"{DATASET}/{split}"
    
    # Get list of all image files in the 'x' subdirectory
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    
    # Process each image with progress bar
    for f in tqdm(filenames, desc=f"Processing {split} split"):
        # Read image and rearrange channels:
        # Creates a 3-channel image where:
        # - Channel 1 (green) becomes channel 0
        # - Channel 0 (red) becomes channel 1
        # - Channel 0 (red) is duplicated as channel 2
        x = tif.imread(join(path, "x", f))
        x = np.dstack((x[..., 1], x[..., 0], x[..., 0]))
        
        # Run Cellpose prediction:
        # channels=[1,2] means we're using:
        # - Channel 1 as nucleus channel
        # - Channel 2 as cell channel
        # diameter=70 sets a fixed cell diameter in pixels
        masks_pred, flows, _ = model.eval(x, channels=[1, 2], diameter=70)
        masks.append(masks_pred)
    
    # Create output directory if it doesn't exist
    os.makedirs(join("/bigdata/casus/MLID/maria/VIRVS_data", "masks"), exist_ok=True)
    
    # Save all masks for current split as numpy array
    np.save(
        join(
            "/bigdata/casus/MLID/maria/VIRVS_data/masks",
            f"masks_cell_hadv_{split}.npy",  # Filename indicates cell masks for HAdV
        ),
        np.array(masks),
    )