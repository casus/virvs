import os
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tifffile as tif
from cellpose import models, utils
from tqdm import tqdm

DATASET = "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed"
model = models.CellposeModel(model_type="cyto3", gpu=True)
tf.random.set_seed(42)

for split in ["train", "val", "test"]:
    masks = []
    path = f"{DATASET}/{split}"
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    for f in tqdm(filenames):
        x = tif.imread(join(path, "x", f))
        x = np.dstack((x[..., 1], x[..., 0], x[..., 0]))

        masks_pred, flows, _ = model.eval(x, channels=[1, 2], diameter=70)
        masks.append(masks_pred)

    os.makedirs(join("/bigdata/casus/MLID/maria/VIRVS_data", "masks"), exist_ok=True)
    np.save(
        join(
            "/bigdata/casus/MLID/maria/VIRVS_data/masks",
            f"masks_cell_hadv_{split}.npy",
        ),
        np.array(masks),
    )
