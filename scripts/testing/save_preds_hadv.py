from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
import tifffile as tif
from tqdm import tqdm
from virvs.architectures.pix2pix import Generator

BASE_PATH = ""
DATASET = ""

WEIGHTS_1CH = ""
WEIGHTS_2CH = ""

tf.random.set_seed(42)

x = []
cellpose_masks = []
filenames = [f for f in listdir(join(DATASET, "x")) if isfile(join(DATASET, "x", f))]
for f in filenames:
    x.append(tif.imread(join(DATASET, "x", f)))
    cellpose_masks.append(np.load(join(DATASET, "masks", f.replace(".tif", ".npy"))))
x = np.array(x)
cellpose_masks = np.array(cellpose_masks)

np.save("cellpose_masks.npy", cellpose_masks)

generator = Generator(2048, [0, 1], 1)
generator.load_weights(f"{BASE_PATH}{WEIGHTS_2CH}")

pred = []
for batch_x in tqdm(x):
    output = np.squeeze(generator(np.expand_dims(batch_x, 0), training=True), 0)
    pred.append(output)

pred = np.array(pred)
np.save(f"preds_2ch.npy", pred)

x = x[:, :, :, 0:1]

generator = Generator(2048, [0], 1)
generator.load_weights(f"{BASE_PATH}{WEIGHTS_1CH}")

pred = []
for batch_x in tqdm(x):
    output = np.squeeze(generator(np.expand_dims(batch_x, 0), training=True), 0)
    pred.append(output)

pred = np.array(pred)
np.save(f"preds_1ch.npy", pred)
