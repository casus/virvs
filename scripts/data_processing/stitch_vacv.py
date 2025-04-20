"""
This script processes and stitches multi-tile, multi-channel time-lapse microscopy data
from the VACVLarge (2channel) dataset. It organizes scattered files into a structured
output with consistent naming, performs image registration and stitching, and saves
the results for downstream analysis.

Key Features:
- Processes 3 timepoints (100, 108, 115) from time-lapse experiments
- Handles 3 fluorescence channels (w1, w2, w3)
- Computes stitching parameters from w3 channel and applies to all channels
- Corrects for tile overlap (10%) during stitching
- Filters out corrupt samples automatically
- Generates composite stitched images for each timepoint and channel

Workflow:
1. For each sample and timepoint:
   a. Identifies all tile images (9 tiles per channel)
   b. Computes registration parameters using w3 channel
   c. Applies same transformation to all channels (w1, w2, w3)
   d. Stitches tiles into composite images
   e. Saves results with standardized naming convention

Output Structure:
- Creates subdirectory for each sample
- Saves stitched images as:
  {sample_identifier}/{channel}_t{timepoint}.tif

Usage:
- Set IMAGES_DIR and OUTPUT_DIR paths
- Run script to process all valid samples
- Use Fiji to merge timepoints/channels as needed

Note: Automatically excludes corrupt samples listed in TO_REMOVE
"""

import os
import re
from pathlib import Path

import numpy as np
import tifffile
from multiview_stitcher import msi_utils, param_utils, registration, spatial_image_utils
from tqdm import tqdm

# Configuration
TIMEPOINTS_TO_USE = [100, 108, 115]  # Timepoints to process
IMAGES_DIR = Path("")  # Root directory containing TimePoint_X subdirectories
OUTPUT_DIR = Path("")  # Output directory for stitched images

# Corrupt samples to exclude 
TO_REMOVE = [
    "121218-AY-JM-VACV-TC_C02",
    "121218-AY-JM-VACV-TC_C07",
    "121218-AY-JM-VACV-TC_D02",
    "121218-AY-JM-VACV-TC_D07",
    "121218-AY-JM-VACV-TC_E02",
    "121218-AY-JM-VACV-TC_E07",
    "121218-AY-JM-VACV-TC_F02",
    "121218-AY-JM-VACV-TC_F07",
]


def main(time_point, sample_identifier):
    """
    Process and stitch images for a specific sample and timepoint.
    
    Args:
        time_point (int): Timepoint to process (e.g., 100)
        sample_identifier (str): Sample ID (e.g., "121218-AY-JM-VACV-TC_A01")
    """
    images_dir = IMAGES_DIR / f"TimePoint_{time_point}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get w3 channel files for registration
    files = get_files(
        dir=images_dir, sample_identifier=sample_identifier, channel_identifier="w3"
    )

    # Create multi-scale images and set initial transformation
    msims = get_msims_from_files(files, channels=[3])
    set_initial_transformation_guess(msims)

    # Perform registration using w3 channel
    params = registration.register(
        msims, reg_channel_index=0, transform_key="affine_manual"
    )
    for msim, param in zip(msims, params):
        msi_utils.set_affine_transform(
            msim,
            param,
            transform_key="affine_registered",
            base_transform_key="affine_manual",
        )

    # Get stitching parameters from registered images
    stitching_params_xy = get_stitching_params_from_msims(msims)

    # Apply stitching to all channels
    channels = ["w1", "w2", "w3"]
    for ch in channels:
        files = get_files(
            dir=images_dir, sample_identifier=sample_identifier, channel_identifier=ch
        )
        stitched_image = stitch_files_by_params(files, stitching_params_xy)
        path = os.path.join(OUTPUT_DIR, sample_identifier)
        os.makedirs(path, exist_ok=True)
        tifffile.imwrite(
            os.path.join(path, ch + f"_t{time_point}.tif"),
            stitched_image,
        )


def get_files(dir, sample_identifier, channel_identifier=None):
    """
    Get sorted list of tile files for a sample and channel.
    
    Args:
        dir (Path): Directory containing TimePoint_X folders
        sample_identifier (str): Sample ID prefix
        channel_identifier (str): Channel identifier (w1/w2/w3)
    
    Returns:
        List[Path]: Sorted list of file paths
    """
    def sort_by_s(fn):
        """Sort helper function by tile number (s1-s9)"""
        s_digit = int(fn.split("/")[-1].split("_s")[1][0])
        return s_digit

    files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.startswith(sample_identifier)
        and f.endswith(".tif")
        and (channel_identifier is None or channel_identifier in f)
    ]
    files.sort(key=sort_by_s)
    files = [Path(fn) for fn in files]
    print(
        f"Found {len(files)} files:\n"
        + "\n".join(["  " + str(fn.resolve()) for fn in files]),
    )
    return files


def get_fp_from_tile_and_channel(file_list, filetile, channel):
    """
    Get file path for specific tile and channel.
    
    Args:
        file_list (List[Path]): List of all files
        filetile (int): Tile number (1-9)
        channel (int): Channel number (1-3)
    
    Returns:
        Path: Matching file path or None
    """
    for fp in file_list:
        filename = fp.name
        match = re.search(r".*_s(\d+)_w(\d).*.tif", filename)
        if match:
            tile_number = int(match.group(1))
            channel_number = int(match.group(2))
            if tile_number == filetile and channel_number == channel:
                return fp
    return None


def get_msims_from_files(files, channels=[1, 2, 3]):
    """
    Create multi-scale images from tile files.
    
    Args:
        files (List[Path]): List of input files
        channels (List[int]): Channels to include
    
    Returns:
        List: Multi-scale image objects
    """
    msims = []
    tiles = np.arange(1, 10)  # 3x3 grid of tiles
    for tile in tiles:
        im_data = np.array(
            [
                tifffile.imread(get_fp_from_tile_and_channel(files, tile, channel))
                for channel in channels
            ]
        )
        sim = spatial_image_utils.get_sim_from_array(
            im_data, dims=("c", "y", "x"), scale={"y": 1, "x": 1}
        )
        msim = msi_utils.get_msim_from_sim(sim, scale_factors=[])
        msims.append(msim)
    return msims


def set_initial_transformation_guess(msims):
    """
    Set initial affine transformation accounting for 10% tile overlap.
    
    Args:
        msims: List of multi-scale images
    """
    overlap = 0.1  # 10% overlap between tiles
    for tile_index, msim in enumerate(msims):
        x_index = tile_index % 3  # Column position (0-2)
        y_index = tile_index // 3  # Row position (0-2)

        tile_extent = (
            spatial_image_utils.get_center_of_sim(msi_utils.get_sim_from_msim(msim)) * 2
        )
        y_extent, x_extent = tile_extent

        # Create affine transform accounting for grid position and overlap
        affine = param_utils.affine_from_translation(
            [y_index * (1 - overlap) * y_extent, x_index * (1 - overlap) * x_extent]
        )

        msi_utils.set_affine_transform(
            msim,
            affine[None],  # one timepoint
            transform_key="affine_manual",
        )


def get_stitching_params_from_msims(msims):
    """
    Extract stitching parameters from registered images.
    
    Args:
        msims: List of registered multi-scale images
    
    Returns:
        np.array: Stitching parameters (9x2 array of x,y offsets)
    """
    params_xy = np.zeros((9, 2), dtype=int)
    for i in range(len(msims)):
        y = msims[i]["scale0/affine_registered"].data[0, 0, 2]
        x = msims[i]["scale0/affine_registered"].data[0, 1, 2]
        params_xy[i, 0] = x.astype(int)
        params_xy[i, 1] = y.astype(int)
    # Normalize to remove negative values
    params_xy[:, 0] -= params_xy[:, 0].min()
    params_xy[:, 1] -= params_xy[:, 1].min()
    return params_xy


def stitch_files_by_params(files, params):
    """
    Stitch tiles together using computed parameters.
    
    Args:
        files (List[Path]): List of input files
        params (np.array): Stitching parameters
    
    Returns:
        np.array: Composite stitched image
    """
    images = [tifffile.imread(f) for f in files]
    size_y, size_x = images[0].shape
    
    # Calculate output dimensions
    stitched_size = (
        params[:, 1].max() + size_y,
        params[:, 0].max() + size_x,
    )

    stitched_image = np.zeros_like(images, shape=stitched_size)
    for i in range(len(images)):
        stitched_image[
            params[i, 1] : params[i, 1] + size_y,
            params[i, 0] : params[i, 0] + size_x,
        ] = images[i]
    return stitched_image


if __name__ == "__main__":
    # Get all unique sample identifiers (first 24 chars of filename)
    samples_list = list(
        set([filename[:24] for filename in os.listdir(IMAGES_DIR / f"TimePoint_1")])
    )
    print("Samples to process:", samples_list)
    
    # Process each sample through all timepoints
    for sample_identifier in tqdm(samples_list):
        if any([x in sample_identifier for x in TO_REMOVE]):
            print("Skipping corrupt sample:", sample_identifier)
            continue
            
        print(f"\nProcessing sample: {sample_identifier}")
        for timepoint in TIMEPOINTS_TO_USE:
            print(f"  Stitching timepoint {timepoint}")
            main(timepoint, sample_identifier)
        print(f"Completed! Results in: {OUTPUT_DIR}/{sample_identifier}")