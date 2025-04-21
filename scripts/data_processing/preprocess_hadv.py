"""
This script processes multi-channel time-lapse microscopy data og HAdV infection (w1, w2, w4 channels)
from multiple timepoints, normalizes the images, and splits them into train/val/test sets.

Key Features:
- Processes multiple wavelength channels (w1, w2, w4) from microscopy data
- Performs channel-specific normalization with predefined min/max values
- Splits data into train/val/test sets with customizable fractions
- Handles timepoint organization in the directory structure
- Saves processed data in standardized format for machine learning

Input Structure:
- Expected directory structure: .../TimePoint_X/*_w[1|2|4].tif
- Each timepoint has corresponding w1, w2, and w4 channel images

Output Structure:
- Creates 'raw' and 'processed' subdirectories
- Processed data organized in split-specific directories (train/val/test)
- Each split contains 'x' (input) and 'gt' (ground truth) subdirectories
"""

import os
from pathlib import Path
import numpy as np
import tifffile as tif
from tqdm import tqdm

RANDOM_SEED = os.environ['RANDOM_SEED']

# Configuration parameters
INPUT_PATH = "/bigdata/casus/MLID/maria/VIRVS_data/HADV/raw"  # Root path to input data
OUTPUT_PATH = f"/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_{RANDOM_SEED}"  # Root path for output data
VAL_FRACTION = 0.2  # Fraction of data for validation set
TEST_FRACTION = 0.1  # Fraction of data for test set

# Set random seed for reproducible data splitting
np.random.seed(RANDOM_SEED)

def sample_norm(x, min, max):
    """Normalize and scale image data to [-1, 1] range based on predefined min/max values.
    
    Args:
        x: Input image data
        min: Predefined minimum value for normalization
        max: Predefined maximum value for normalization
    
    Returns:
        Normalized image data in [-1, 1] range
    """
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1

def read_tiff(path):
    """Read TIFF file and handle 4D stacks by selecting first channel if needed.
    
    Args:
        path: Path to TIFF file
    
    Returns:
        Image data as numpy array
    """
    im_stack = tif.imread(path)
    if len(im_stack.shape) == 4:  # If 4D stack (T,C,X,Y)
        im_stack = im_stack[:, 0]  # Take first channel
    return im_stack

# Get all w1 channel paths for TimePoint 1 (used as reference)
paths_w1 = list(Path(INPUT_PATH).glob("**/TimePoint_" + str(1) + "/*_w1.tif"))
n_sequences = len(paths_w1)
sequence_length = 1  # Currently processing single timepoint

# Initialize array to store all paths
paths = np.empty((n_sequences, sequence_length), dtype=object)

# Process specific timepoint (hardcoded to 48 for this script)
for tdx in tqdm([48], desc="Processing timepoints"):
    for idx in tqdm(range(n_sequences), desc="Organizing paths"):
        # Generate paths for current timepoint by replacing TimePoint_1 with target timepoint
        w1_path_t1 = str(paths_w1[idx])
        w1_path = w1_path_t1.replace("TimePoint_1", "TimePoint_" + str(tdx + 1))
        paths[idx, tdx - 48] = Path(w1_path)

# First pass: Save raw images (all channels) to output directory
for paths_row in tqdm(paths, desc="Saving raw images"):
    for path in paths_row:
        filename = path.name
        out_path = os.path.join(OUTPUT_PATH, "raw", filename)
        
        # Create output directory and save all channels
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tif.imwrite(out_path, tif.imread(path))  # w1 channel
        tif.imwrite(out_path.replace("w1", "w2"), tif.imread(str(path).replace("w1", "w2")))  # w2 channel
        tif.imwrite(out_path.replace("w1", "w4"), tif.imread(str(path).replace("w1", "w4")))  # w4 channel

# Shuffle paths for random split
indices = np.arange(len(paths))
np.random.shuffle(indices)
paths = paths[indices]

# Calculate split indices
val_split_idx = int(len(paths) * (1 - VAL_FRACTION - TEST_FRACTION))
test_split_idx = int(len(paths) * (1 - TEST_FRACTION))

# Create splits
paths_train = paths[:val_split_idx].flatten()
paths_val = paths[val_split_idx:test_split_idx].flatten()
paths_test = paths[test_split_idx:].flatten()

print("Data split sizes:")
print(paths_train.shape, paths_val.shape, paths_test.shape)

# Process and save normalized data for each split
for split, paths in zip(["train", "val", "test"], [paths_train, paths_val, paths_test]):
    for path in tqdm(paths, desc=f"Processing {split} split"):
        # Read and normalize each channel
        w1_ch = np.expand_dims(tif.imread(path), -1) / 65535.0  # 16-bit to [0,1]
        w2_path = Path(str(path).replace("w1", "w2"))
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0
        w4_path = Path(str(path).replace("w1", "w4"))
        w4_ch = np.expand_dims(tif.imread(w4_path), -1) / 65535.0

        # Apply channel-specific normalization with normalization constants chosen by hand
        w1_ch = sample_norm(w1_ch, min=0.0035, max=0.0178)
        w2_ch = sample_norm(w2_ch, min=0.0104, max=0.0741)
        w4_ch = sample_norm(w4_ch, min=0.032, max=0.4089)

        # Prepare output filenames
        filename = path.name[: path.name.rindex("_")]
        parent_dir = path.parent.name
        timepoint = parent_dir.split("_")[-1]
        new_filename = f"{filename}_{timepoint}.tif"

        # Create output directories
        subdir = "processed"
        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)

        # Save processed data
        # x: concatenated w1 and w4 channels (input features)
        # gt: w2 channel (ground truth)
        tif.imwrite(out_path_x, np.concatenate([w1_ch, w4_ch], axis=-1))
        tif.imwrite(out_path_gt, w2_ch)