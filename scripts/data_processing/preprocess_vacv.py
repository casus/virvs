import os
from pathlib import Path

import numpy as np
import tifffile as tif
from tqdm import tqdm

TIMEPOINTS = [100, 108, 115]

INPUT_PATH = ""
OUTPUT_PATH = ""


VAL_FRACTION = 0.2
TEST_FRACTION = 0.1


def center_crop(x, crop_size_y, crop_size_x):
    x_center = x.shape[1] // 2
    y_center = x.shape[0] // 2
    return x[
        y_center - crop_size_y // 2 : y_center + crop_size_y // 2,
        x_center - crop_size_x // 2 : x_center + crop_size_x // 2,
    ]


def sample_norm(x, min, max):
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1


np.random.seed(42)

paths_w1_t1 = list(Path(INPUT_PATH).glob(f"**/w1_t{TIMEPOINTS[0]}.tif"))
n_sequences = len(paths_w1_t1)

paths = np.empty((n_sequences, len(TIMEPOINTS)), dtype=object)
for tdx in tqdm(TIMEPOINTS):
    for idx in tqdm(range(n_sequences)):

        w1_path_t1 = str(paths_w1_t1[idx])
        w1_path = w1_path_t1.replace(f"_t{TIMEPOINTS[0]}", f"_t{tdx}")
        paths[idx, TIMEPOINTS.index(tdx)] = w1_path

indices = np.arange(len(paths))
np.random.shuffle(indices)
paths = paths[indices]

val_split_idx = int(len(paths) * (1 - VAL_FRACTION - TEST_FRACTION))
test_split_idx = int(len(paths) * (1 - TEST_FRACTION))

paths_train = paths[:val_split_idx].flatten()
paths_val = paths[val_split_idx:test_split_idx].flatten()
paths_test = paths[test_split_idx:].flatten()

print("Saving data")
print(paths_train.shape, paths_val.shape, paths_test.shape)


for split, paths in zip(["train", "val", "test"], [paths_train, paths_val, paths_test]):
    for path in tqdm(paths):
        w2_path = Path(str(path).replace("w1", "w2"))
        w3_path = Path(str(path).replace("w1", "w3"))
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0
        w3_ch = np.expand_dims(tif.imread(w3_path), -1) / 65535.0

        w2_ch = sample_norm(
            center_crop(
                w2_ch,
                5948,
                6048,
            ),
            min=0.0067,
            max=1.0,
        )

        w3_ch = sample_norm(
            center_crop(
                w3_ch,
                5948,
                6048,
            ),
            min=0.4956,
            max=1.0,
        )

        filename = w2_path.name
        parent_dir = w2_path.parent.name

        channel, timepoint = filename.split("_")
        timepoint = timepoint.replace(".tif", "")

        new_filename = f"{parent_dir}_{timepoint}.tif"
        subdir = "processed"

        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)
        tif.imwrite(
            out_path_x,
            w3_ch,
        )
        tif.imwrite(
            out_path_gt,
            w2_ch,
        )
