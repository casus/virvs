import os
from pathlib import Path

import numpy as np
import tifffile as tif
from tqdm import tqdm

VAL_FRACTION = 0.2
TEST_FRACTION = 0.1

np.random.seed(42)

BASE_PATH = ""
SUFFIXES = [
    "3-Screen/Data_UZH/Screen",
    "2-ZPlates/Data_UZH/Za",
    "2-ZPlates/Data_UZH/Zb",
    "1-prePlates/Data_UZH/preZ",
]

OUTPUT_PATH = ""


def sample_norm(x, min, max):
    n_x = np.clip((x - min) / (max - min), 0, 1)
    return n_x * 2 - 1


HSV_percentiles = {
    "w1": {"min": 0.0074, "max": 0.2907},
    "w2": {"min": 0.0033, "max": 0.1926},
}
RV_percentiles = {
    "w1": {"min": 0.0086, "max": 0.2193},
    "w2": {"min": 0.0082, "max": 0.1712},
}
IAV_percentiles = {
    "w1": {"min": 0.011, "max": 0.2156},
    "w2": {"min": 0.0091, "max": 0.1769},
}

percentiles = IAV_percentiles

paths = []
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]
# Walk through the root folder

for suffix in SUFFIXES:
    root_path = BASE_PATH + suffix
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            # Check if the file is an image by its extension
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in image_extensions:
                # Check if the filename contains any of the desired substrings
                if "3-Screen" in root_path:
                    # For 3-Screen we only want to use first 2 columns (with 1 or 2 in the name)
                    if any(
                        f"{letter}{i:02d}" in filename
                        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        for i in range(1, 3)
                    ):

                        file_path = os.path.join(dirpath, filename)
                        paths.append(Path(file_path.replace("w2", "w1")))

                else:
                    if any(
                        f"{letter}{i:02d}" in filename
                        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        for i in range(1, 13)
                    ):
                        file_path = os.path.join(dirpath, filename)
                        paths.append(Path(file_path.replace("w2", "w1")))

for path in tqdm(paths):
    filename = path.name
    out_path = os.path.join(OUTPUT_PATH, "raw", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tif.imwrite(out_path, tif.imread(path))
    tif.imwrite(out_path.replace("w1", "w2"), tif.imread(str(path).replace("w1", "w2")))


paths = np.unique(np.array(paths))
indices = np.arange(len(paths))
np.random.shuffle(indices)
paths = paths[indices]

val_split_idx = int(len(paths) * (1 - VAL_FRACTION - TEST_FRACTION))
test_split_idx = int(len(paths) * (1 - TEST_FRACTION))

paths_train = paths[:val_split_idx]
paths_val = paths[val_split_idx:test_split_idx]
paths_test = paths[test_split_idx:]

print("Saving data")
print(paths_train.shape, paths_val.shape, paths_test.shape)

for split, paths in zip(["train", "val", "test"], [paths_train, paths_val, paths_test]):
    for path in tqdm(paths):

        w1_ch = np.expand_dims(tif.imread(path), -1) / 65535.0

        w2_path = Path(str(path).replace("w1", "w2"))
        w2_ch = np.expand_dims(tif.imread(w2_path), -1) / 65535.0

        w1_ch = sample_norm(
            w1_ch,
            min=percentiles["w1"]["min"],
            max=percentiles["w1"]["max"],
        )
        w2_ch = sample_norm(
            w2_ch,
            min=percentiles["w2"]["min"],
            max=percentiles["w2"]["max"],
        )

        new_filename = path.name[: path.name.rindex("_")] + ".tif"
        subdir = "processed"

        out_path_x = os.path.join(OUTPUT_PATH, subdir, split, "x", new_filename)
        out_path_gt = os.path.join(OUTPUT_PATH, subdir, split, "gt", new_filename)
        os.makedirs(os.path.dirname(out_path_x), exist_ok=True)
        os.makedirs(os.path.dirname(out_path_gt), exist_ok=True)

        tif.imwrite(
            out_path_x,
            w1_ch,
        )
        tif.imwrite(
            out_path_gt,
            w2_ch,
        )
