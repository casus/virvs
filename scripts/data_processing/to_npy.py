"""
This script prepares multi-class image data for training, validation, and testing
by reading `.tif` microscopy images and storing them in `.npy` format for efficient access.

Key Features:
- Loads paired input (x) and ground truth (y) microscopy images from TIFF files
- Processes three predefined data splits: train, val, and test
- Converts image sequences into NumPy arrays for deep learning pipelines
- Creates JSON index files mapping filenames to array indices
- Uses `tqdm` for progress visualization during processing

Data Flow:
- Input: Directory structure with `x/` (input images) and `gt/` (ground truth images)
- Processing: Image loading → Aggregation into lists → Conversion to NumPy arrays
- Output: `x.npy`, `y.npy`, and `filename_to_index.json` files stored in each split directory
"""

import json
from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile as tif
from tqdm import tqdm
import os
# Root path to preprocessed microscopy data organized by split
RANDOM_SEED = os.environ['RANDOM_SEED']
VIRUS = os.environ['VIRUS']

PATH = f"/bigdata/casus/MLID/maria/VIRVS_data/{VIRUS}/processed_{RANDOM_SEED}"

# Iterate over dataset splits
for split in tqdm(["train", "val", "test"]):
    path = join(PATH, split)
    x = []  # List to hold input images
    y = []  # List to hold ground truth images
    filename_index = {}  # Dict to map filenames to array indices

    # Gather all filenames from the input directory
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]

    # Load input and ground truth image pairs
    for idx, f in enumerate(tqdm(filenames)):
        x.append(tif.imread(join(path, "x", f)))      # Load input image
        y.append(tif.imread(join(path, "gt", f)))     # Load ground truth image
        filename_index[f] = idx  # Record filename-to-index mapping

    # Save processed data as NumPy arrays
    np.save(join(path, f"x.npy"), np.array(x))
    np.save(join(path, f"y.npy"), np.array(y))
    
    # Save filename-to-index mapping as JSON
    with open(join(path, "filename_to_index.json"), 'w') as f:
        json.dump(filename_index, f, indent=4)