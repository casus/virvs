from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tifffile as tif
from tqdm import tqdm
from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader

BASE_PATH = ""
DATASET = ""

WEIGHTS = ""
virus = "vacv"

tf.random.set_seed(42)

if "vacv" not in virus:
    size = 2048
else:
    size = 5888
# crop center if not fitting
dataloader = NpyDataloader(
    path=DATASET,
    im_size=size,
    random_jitter=False,
    crop_type="center",
    ch_in=[0],
)
x = dataloader._x

generator = Generator(size, [0], 1)
generator.load_weights(f"{BASE_PATH}{WEIGHTS}")

pred = []
for batch_x in tqdm(x):
    output = np.squeeze(generator(np.expand_dims(batch_x, 0), training=True), 0)
    pred.append(output)

pred = np.array(pred)
np.save(f"preds.npy", pred)
