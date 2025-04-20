"""
This script prepares multi-class image data for training, validation, and testing
by reading `.tif` microscopy images and storing them in `.npy` format for efficient access.

Key Features:
- Loads paired input (x) and ground truth (y) microscopy images from TIFF files
- Processes three predefined data splits: train, val, and test
- Converts image sequences into NumPy arrays for deep learning pipelines
- Uses `tqdm` for progress visualization during processing

Data Flow:
- Input: Directory structure with `x/` (input images) and `gt/` (ground truth images)
- Processing: Image loading → Aggregation into lists → Conversion to NumPy arrays
- Output: `x.npy` and `y.npy` files stored in each split directory
"""

from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile as tif
from tqdm import tqdm

# Root path to preprocessed microscopy data organized by split
PATH = "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed"

# Iterate over dataset splits
for split in tqdm(["train", "val", "test"]):
    path = join(PATH, split)
    x = []  # List to hold input images
    y = []  # List to hold ground truth images

    # Gather all filenames from the input directory
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]

    # Load input and ground truth image pairs
    for f in tqdm(filenames):
        x.append(tif.imread(join(path, "x", f)))      # Load input image
        y.append(tif.imread(join(path, "gt", f)))     # Load ground truth image

    # Save processed data as NumPy arrays
    np.save(join(path, f"x.npy"), np.array(x))
    np.save(join(path, f"y.npy"), np.array(y))
