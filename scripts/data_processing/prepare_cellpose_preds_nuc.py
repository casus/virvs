import os
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile as tif
from cellpose import models, utils
from tqdm import tqdm

DATASETS = {
    "hadv": "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed/test",
    "iav": "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed/test",
    "hsv": "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed/test",
    "rv": "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed/test",
}


DIAMETERS = {
    "hadv": None,
    "hsv": 7,
    "iav": 7,
    "rv": 7,
}

tf.random.set_seed(42)
model = models.CellposeModel(model_type="cyto3", gpu=True)

for virus in DATASETS.keys():
    masks = []
    path = f"{DATASETS[virus]}"
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    for f in tqdm(filenames):
        x = tif.imread(join(path, "x", f))[..., 0]
        masks_pred, flows, _ = model.eval(x, channels=[0, 0], diameter=DIAMETERS[virus])
        masks.append(masks_pred)
    os.makedirs(join("/bigdata/casus/MLID/maria/VIRVS_data", "masks"), exist_ok=True)
    np.save(
        join(
            "/bigdata/casus/MLID/maria/VIRVS_data/masks",
            f"masks_nuc_{virus}_test.npy",
        ),
        np.array(masks),
    )
