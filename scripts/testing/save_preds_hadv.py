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

WEIGHTS_1CH = ""
WEIGHTS_2CH = ""

tf.random.set_seed(42)

dataloader = NpyDataloader(
    path=DATASET,
    im_size=2048,
    random_jitter=False,
    ch_in=[0, 1],
)

generator = Generator(2048, [0, 1], 1)
generator.load_weights(f"{BASE_PATH}{WEIGHTS_2CH}")

pred = []
for sample in tqdm(dataloader):
    x, y = sample
    output = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)
    pred.append(output)

pred = np.array(pred)
np.save(f"preds_2ch.npy", pred)

dataloader = NpyDataloader(
    path=DATASET,
    im_size=2048,
    random_jitter=False,
    ch_in=[0],
)

generator = Generator(2048, [0], 1)
generator.load_weights(f"{BASE_PATH}{WEIGHTS_1CH}")

pred = []
for sample in tqdm(dataloader):
    x, y = sample
    output = np.squeeze(generator(np.expand_dims(x, 0), training=True), 0)
    pred.append(output)

pred = np.array(pred)
np.save(f"preds_1ch.npy", pred)
