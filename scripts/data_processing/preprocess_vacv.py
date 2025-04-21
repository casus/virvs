"""
This script processes multi-channel time-lapse microscopy data of VACV infection across specific timepoints,
performs center cropping, normalization, and splits the data into train/val/test sets.

Key Features:
- Processes multiple timepoints (100, 108, 115) from time-lapse experiments
- Handles three fluorescence channels (w1, w2, w3)
- Performs precise center cropping of large microscopy images
- Applies channel-specific normalization with predefined min/max values
- Implements structured train/val/test splitting
- Maintains temporal information in output filenames

Data Flow:
- Input: w1, w2, w3 channel images organized by timepoint
- Processing: Center cropping → Normalization → Channel selection
- Output: w3 as input features, w2 as ground truth
"""

import os
from pathlib import Path
import numpy as np
import tifffile as tif
from tqdm import tqdm

# Configuration parameters
RANDOM_SEED = int(os.environ['RANDOM_SEED'])


TIMEPOINTS = [100, 108, 115]  # Specific timepoints to process
INPUT_PATH = "/bigdata/casus/MLID/maria/VIRVS_data/VACV/raw" # Root path to input data
OUTPUT_PATH = f"/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed_{RANDOM_SEED}"  # Root path for output data
VAL_FRACTION = 0.2  # Fraction of data for validation set
TEST_FRACTION = 0.1  # Fraction of data for test set
np.random.seed(RANDOM_SEED)  # Seed for reproducible random splitting

def center_crop(x, crop_size_y, crop_size_x):
    """Center crop an image to specified dimensions.
    
    Args:
        x: Input image (height, width) or (height, width, channels)
        crop_size_y: Desired output height
        crop_size_x: Desired output width
    
    Returns:
        Center-cropped image of size (crop_size_y, crop_size_x)
    """
    x_center = x.shape[1] // 2
    y_center = x.shape[0] // 2
    return x[
        y_center - crop_size_y // 2 : y_center + crop_size_y // 2,
        x_center - crop_size_x // 2 : x_center + crop_size_x // 2,
    ]

def sample_norm(x, min, max):
    """Normalize and scale image data to [-1, 1] range.
    
    Args:
        x: Input image data
        min: Minimum value for normalization
        max: Maximum value for normalization
    
    Returns:
        Normalized image data in [-1, 1] range
    """
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1

# Get all w1 channel paths for first timepoint (reference)
paths_w1_t1 = list(Path(INPUT_PATH).glob(f"**/w1_t{TIMEPOINTS[0]}.tif"))
n_sequences = len(paths_w1_t1)

# Initialize array to store all paths (sequences × timepoints)
paths = np.empty((n_sequences, len(TIMEPOINTS)), dtype=object)

# Organize paths by timepoint
for tdx in tqdm(TIMEPOINTS, desc="Organizing timepoints"):
    for idx in tqdm(range(n_sequences), desc="Building path matrix", leave=False):
        # Generate paths for current timepoint by replacing timepoint in filename
        w1_path_t1 = str(paths_w1_t1[idx])
        w1_path = w1_path_t1.replace(f"_t{TIMEPOINTS[0]}", f"_t{tdx}")
        paths[idx, TIMEPOINTS.index(tdx)] = w1_path

# Shuffle sequences while maintaining timepoint groupings
indices = np.arange(len(paths))
np.random.shuffle(indices)
paths = paths[indices]

# Calculate split indices
val_split_idx = int(len(paths) * (1 - VAL_FRACTION - TEST_FRACTION))
test_split_idx = int(len(paths) * (1 - TEST_FRACTION))

# Create dataset splits
paths_train = paths[:val_split_idx].flatten()
paths_val = paths[val_split_idx:test_split_idx].flatten()
paths_test = paths[test_split_idx:].flatten()

print("Dataset split sizes:")
print(f"Train: {paths_train.shape[0]}, Val: {paths_val.shape[0]}, Test: {paths_test.shape[0]}")

# Process and save data for each split
for split, paths in zip(["train", "val", "test"], [paths_train, paths_val, paths_test]):
    for path in tqdm(paths, desc=f"Processing {split} split"):
        # Get corresponding w2 and w3 channel paths
        w2_path = Path(str(path).replace("w1", "w2"))
        w3_path = Path(str(path).replace("w1", "w3"))
        
        # Read and normalize channels (16-bit to [0,1])
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0
        w3_ch = np.expand_dims(tif.imread(w3_path), -1) / 65535.0

        # Center crop and normalize w2 channel (ground truth)
        w2_ch = sample_norm(
            center_crop(w2_ch, 5948, 6048),  # Large crop size for high-res images
            min=0.0067,  # Predefined normalization parameters
            max=1.0
        )

        # Center crop and normalize w3 channel (input features)
        w3_ch = sample_norm(
            center_crop(w3_ch, 5948, 6048),
            min=0.4956,
            max=1.0
        )

        # Prepare output filename with parent directory and timepoint
        filename = w2_path.name
        parent_dir = w2_path.parent.name
        channel, timepoint = filename.split("_")
        timepoint = timepoint.replace(".tif", "")
        new_filename = f"{parent_dir}_{timepoint}.tif"

        # Create output directories
        subdir = "processed"
        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)

        # Save processed data
        # x: w3 channel (input features)
        # gt: w2 channel (ground truth)
        tif.imwrite(out_path_x, w3_ch)
        tif.imwrite(out_path_gt, w2_ch)