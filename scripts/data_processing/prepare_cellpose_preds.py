import os
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tifffile as tif
from cellpose import models, utils
from tqdm import tqdm

DATASET = ""
model = models.CellposeModel(model_type="cyto3", gpu=True)
tf.random.set_seed(42)

for split in ["train", "val", "test"]:
    path = f"{DATASET}/{split}"
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    for f in tqdm(filenames):
        x = tif.imread(join(path, "x", f))[..., 1]
        masks_pred, flows, _ = model.eval(x, channels=[0, 0], diameter=70)
        outlines = utils.masks_to_outlines(masks_pred)

        os.makedirs(join(path, "masks"), exist_ok=True)
        np.save(join(path, "masks", f.replace(".tif", ".npy")), masks_pred)
