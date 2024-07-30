"""Script to collect and stitch time-lapse videos for the 2channel dataset (aka VACVLarge).

Purpose of this script is to store all files scattered in different folders and
all saved with a convenient name in one folder.

For each frame, the stitching parameters for w3 channel is computed. Then, the
same parameters are used for the other two channels as well. This will result in
3 tiff files for each frame each channel. The same is applied for the other
frames as well. Then saved and compressed into a single file with proper names
to be downloaded from the server. Then on a local machine with Fiji, one can
merge any of the channles and also any of the frames needed to get the time-lapse
images.
"""

import os
import re
from pathlib import Path

import numpy as np
import tifffile
from multiview_stitcher import msi_utils, param_utils, registration, spatial_image_utils
from tqdm import tqdm

TIMEPOINTS_TO_USE = [100, 108, 115]
IMAGES_DIR = Path("")
OUTPUT_DIR = Path("")

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
    images_dir = IMAGES_DIR / f"TimePoint_{time_point}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get the 9 files we need to stitch
    files = get_files(
        dir=images_dir, sample_identifier=sample_identifier, channel_identifier="w3"
    )

    msims = get_msims_from_files(files, channels=[3])

    set_initial_transformation_guess(msims)

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

    # Use ch3 parameters for the other channels, stitch them and save them in a
    # folder for download
    stitching_params_xy = get_stitching_params_from_msims(msims)

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
    """Get all files in a directory.

    Args:
        dir (str): Directory to look for files.
        sample_identifier (str): Sample identifier to look for.
        channel_identifier (str): Channel identifier to look for.

    Returns:
        list: Sorted list of files.
    """

    def sort_by_s(fn):
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
    for fp in file_list:
        filename = fp.name
        # Extract tile number and channel number from the filename
        match = re.search(r".*_s(\d+)_w(\d).*.tif", filename)
        if match:
            tile_number = int(match.group(1))
            channel_number = int(match.group(2))

            # Check if the extracted tile and channel numbers match the provided ones
            if tile_number == filetile and channel_number == channel:
                return fp

    # If no matching filename found, return None
    return None


def get_msims_from_files(files, channels=[1, 2, 3]):
    msims = []
    tiles = np.arange(1, 10)
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
    overlap = 0.1
    for tile_index, msim in enumerate(msims):
        x_index = tile_index % 3
        y_index = tile_index // 3

        tile_extent = (
            spatial_image_utils.get_center_of_sim(msi_utils.get_sim_from_msim(msim)) * 2
        )
        y_extent, x_extent = tile_extent

        affine = param_utils.affine_from_translation(
            [y_index * (1 - overlap) * y_extent, x_index * (1 - overlap) * x_extent]
        )

        msi_utils.set_affine_transform(
            msim,
            affine[None],  # one tp
            transform_key="affine_manual",
        )


def get_stitching_params_from_msims(msims):
    params_xy = np.zeros((9, 2), dtype=int)
    for i in range(len(msims)):
        y = msims[i]["scale0/affine_registered"].data[0, 0, 2]
        x = msims[i]["scale0/affine_registered"].data[0, 1, 2]
        params_xy[i, 0] = x.astype(int)
        params_xy[i, 1] = y.astype(int)
    # Since these params can contain negatives and invalid values, we modify here
    params_xy[:, 0] -= params_xy[:, 0].min()
    params_xy[:, 1] -= params_xy[:, 1].min()
    return params_xy


def stitch_files_by_params(files, params):
    images = [tifffile.imread(f) for f in files]

    size_y, size_x = images[0].shape
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

    samples_list = list(
        set([filename[:24] for filename in os.listdir(IMAGES_DIR / f"TimePoint_1")])
    )
    print("Samples list", samples_list)
    for sample_identifier in tqdm(samples_list):
        if any([x in sample_identifier for x in TO_REMOVE]):
            print("Ignoring sample ", sample_identifier)
            continue
        else:
            store_path = OUTPUT_DIR / sample_identifier
            timepoints = TIMEPOINTS_TO_USE

            print(f"Stitching sample: {sample_identifier}")
            for i in timepoints:
                print(f"Processing timepoint {i}")
                main(i, sample_identifier)
            print(f"Done! Results saved in: {OUTPUT_DIR}/{sample_identifier}")
