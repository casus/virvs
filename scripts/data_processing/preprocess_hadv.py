import os
from pathlib import Path

import numpy as np
import tifffile as tif
from tqdm import tqdm

INPUT_PATH = ""
OUTPUT_PATH = ""

VAL_FRACTION = 0.2
TEST_FRACTION = 0.1

np.random.seed(42)


def sample_norm(x, min, max):
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1


def read_tiff(path):
    im_stack = tif.imread(path)
    if len(im_stack.shape) == 4:
        im_stack = im_stack[:, 0]

    return im_stack


paths_w1 = list(Path(INPUT_PATH).glob("**/TimePoint_" + str(1) + "/*_w1.tif"))
n_sequences = len(paths_w1)
sequence_length = 1

paths = np.empty((n_sequences, sequence_length), dtype=object)

for tdx in tqdm([48]):

    for idx in tqdm(range(n_sequences)):

        w1_path_t1 = str(paths_w1[idx])
        w1_path = w1_path_t1.replace("TimePoint_1", "TimePoint_" + str(tdx + 1))
        paths[idx, tdx - 48] = Path(w1_path)

for paths_row in tqdm(paths):
    for path in paths_row:
        filename = path.name
        out_path = os.path.join(OUTPUT_PATH, "raw", filename)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tif.imwrite(out_path, tif.imread(path))
        tif.imwrite(
            out_path.replace("w1", "w2"), tif.imread(str(path).replace("w1", "w2"))
        )
        tif.imwrite(
            out_path.replace("w1", "w4"), tif.imread(str(path).replace("w1", "w4"))
        )

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
        w1_ch = np.expand_dims(tif.imread(path), -1) / 65535.0

        w2_path = Path(str(path).replace("w1", "w2"))
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0

        w4_path = Path(str(path).replace("w1", "w4"))
        w4_ch = np.expand_dims(tif.imread(w4_path), -1) / 65535.0

        w1_ch = sample_norm(
            w1_ch,
            min=0.0035,
            max=0.0178,
        )
        w2_ch = sample_norm(
            w2_ch,
            min=0.0104,
            max=0.0741,
        )
        w4_ch = sample_norm(
            w4_ch,
            min=0.032,
            max=0.4089,
        )

        filename = path.name[: path.name.rindex("_")]
        parent_dir = path.parent.name

        timepoint = parent_dir.split("_")[-1]

        new_filename = f"{filename}_{timepoint}.tif"

        subdir = "processed"

        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)

        tif.imwrite(
            out_path_x,
            np.concatenate([w1_ch, w4_ch], axis=-1),
        )
        tif.imwrite(
            out_path_gt,
            w2_ch,
        )

