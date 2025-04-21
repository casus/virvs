"""
This script processes high-content screening microscopy data from multiple viruses (HSV, RV, IAV).
performs channel-specific normalization, and splits the data into train/val/test sets.

Key Features:
- Processes multi-plate HCS data with different plate configurations
- Handles w1 and w2 fluorescence channels
- Applies virus-specific normalization based on predefined percentiles
- Supports multiple virus datasets (HSV, RV, IAV)
- Implements structured train/val/test splitting
- Maintains original file organization while creating processed outputs

Directory Structure:
- Input: Organized by plate type (3-Screen, 2-ZPlates, 1-prePlates)
- Output: Creates 'raw' and 'processed' directories with split subfolders
- Processed data organized in 'x' (input) and 'gt' (ground truth) subdirectories
"""

import os
from pathlib import Path
import numpy as np
import tifffile as tif
from tqdm import tqdm

RANDOM_SEED = int(os.environ['RANDOM_SEED'])

VIRUS = os.environ['VIRUS']
# Configuration parameters
VAL_FRACTION = 0.2  # Fraction of data for validation set
TEST_FRACTION = 0.1  # Fraction of data for test set
np.random.seed(RANDOM_SEED)   # Seed for reproducible random splitting

# Path configuration
BASE_PATH = f"/bigdata/casus/MLID/maria/VIRVS_data/{VIRUS}/raw/"  # Root path to input data
OUTPUT_PATH = f"/bigdata/casus/MLID/maria/VIRVS_data/{VIRUS}/"  # Root path for output data

# Plate directory suffixes to process
SUFFIXES = [
    "3-Screen/Data_UZH/Screen",  
    "2-ZPlates/Data_UZH/Za",   
    "2-ZPlates/Data_UZH/Zb",    
    "1-prePlates/Data_UZH/preZ", 
]

def sample_norm(x, min, max):
    """Normalize and scale image data to [-1, 1] range based on predefined percentiles.
    
    Args:
        x: Input image data (normalized to [0,1])
        min: Predefined minimum value for this channel
        max: Predefined maximum value for this channel
    
    Returns:
        Normalized image data in [-1, 1] range
    """
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1

# Channel-specific normalization parameters for different viruses
percentiles_dict = {"HSV":{
    "w1": {"min": 0.0074, "max": 0.2907},  
    "w2": {"min": 0.0033, "max": 0.1926}, 
},
"RV":{
    "w1": {"min": 0.0086, "max": 0.2193},
    "w2": {"min": 0.0082, "max": 0.1712},
},
"IAV":{
    "w1": {"min": 0.011, "max": 0.2156},
    "w2": {"min": 0.0091, "max": 0.1769},
}
}

percentiles = percentiles_dict[VIRUS]

# # Initialize list to store all image paths
# paths = []
# image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]

# # Walk through directory structure to find all relevant images
# for suffix in SUFFIXES:
#     root_path = BASE_PATH + suffix
#     for dirpath, _, filenames in os.walk(root_path):
#         for filename in filenames:
#             file_extension = os.path.splitext(filename)[1].lower()
            
#             # Only process supported image files
#             if file_extension in image_extensions:
#                 # Different plate types have different well selection criteria
#                 if "3-Screen" in root_path:
#                     # For screen plates, only use first 2 columns (wells 1-2)
#                     if any(f"{letter}{i:02d}" in filename
#                          for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#                          for i in range(1, 3)):
#                         file_path = os.path.join(dirpath, filename)
#                         # Store w1 path but we'll process both channels
#                         paths.append(Path(file_path.replace("w2", "w1")))
#                 else:
#                     # For other plates, use all 12 columns (wells 1-12)
#                     if any(f"{letter}{i:02d}" in filename
#                          for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#                          for i in range(1, 13)):
#                         file_path = os.path.join(dirpath, filename)
#                         paths.append(Path(file_path.replace("w2", "w1")))

# # Save raw images (both channels) to output directory
# for path in tqdm(paths, desc="Saving raw images"):
#     filename = path.name
#     out_path = os.path.join(OUTPUT_PATH, "raw", filename)
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     # Save both channels
#     tif.imwrite(out_path, tif.imread(path))  # w1 channel
#     tif.imwrite(out_path.replace("w1", "w2"),  # w2 channel
#                tif.imread(str(path).replace("w1", "w2")))

def get_file_paths(virus: str) -> list[str]:
    """Returns a list of absolute paths to all files in the virus raw data directory."""
    base_dir = Path(f"/bigdata/casus/MLID/maria/VIRVS_data/{virus}/raw/")
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    file_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            # should only use w1
            if "w1" in file:
                file_paths.append(Path(root) / file)
    
    return sorted(file_paths)

# Prepare for data splitting - remove duplicates and shuffle
paths = get_file_paths(VIRUS)
paths = np.unique(np.array(paths))

indices = np.arange(len(paths))
np.random.shuffle(indices)
paths = paths[indices]

# Calculate split indices
val_split_idx = int(len(paths) * (1 - VAL_FRACTION - TEST_FRACTION))
test_split_idx = int(len(paths) * (1 - TEST_FRACTION))

# Create dataset splits
paths_train = paths[:val_split_idx]
paths_val = paths[val_split_idx:test_split_idx]
paths_test = paths[test_split_idx:]

print("Dataset split sizes:")
print(f"Train: {paths_train.shape[0]}, Val: {paths_val.shape[0]}, Test: {paths_test.shape[0]}")

# Process and save normalized data for each split
for split, paths in zip(["train", "val", "test"], [paths_train, paths_val, paths_test]):
    for path in tqdm(paths, desc=f"Processing {split} split"):
        # Read and normalize w1 channel (16-bit to [0,1] then to [-1,1])
        w1_ch = np.expand_dims(tif.imread(path), -1) / 65535.0
        w1_ch = sample_norm(w1_ch,
                           min=percentiles["w1"]["min"],
                           max=percentiles["w1"]["max"])

        # Read and normalize w2 channel
        w2_path = Path(str(path).replace("w1", "w2"))
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0
        w2_ch = sample_norm(w2_ch,
                           min=percentiles["w2"]["min"],
                           max=percentiles["w2"]["max"])

        # Prepare output filename (remove trailing _ and add .tif)
        new_filename = path.name[: path.name.rindex("_")] + ".tif"
        subdir = f"processed_{RANDOM_SEED}"

        # Create output paths
        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)

        # Save processed data
        # x: w1 channel (input features)
        # gt: w2 channel (ground truth)
        tif.imwrite(out_path_x, w1_ch)
        tif.imwrite(out_path_gt, w2_ch)