from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from virvs.architectures.pix2pix import Generator
from virvs.data.npy_dataloader import NpyDataloader

VIRUS = ""
DATASET_PATH = ""
WEIGHTS_PIX2PIX = ""
WEIGHTS_UNET = ""

if VIRUS == "vacv":
    size = 5888
else:
    size = 2048

dataloader = NpyDataloader(
    path=DATASET_PATH,
    im_size=size,
    random_jitter=False,
    crop_type="center",
    ch_in=[0],
)

x, y = dataloader[0]

generator = Generator(size, ch_in=[0], ch_out=1)
generator.load_weights(WEIGHTS_PIX2PIX)
pred = generator(np.expand_dims(x, 0), training=True)

plt.axis("off")
plt.title("")
plt.imsave(f"{VIRUS}_pix2pix_prediction.svg", np.squeeze(pred), vmin=-1, vmax=1)

generator = Generator(size, ch_in=[0], ch_out=1)
generator.load_weights(WEIGHTS_UNET)
pred = generator(np.expand_dims(x, 0), training=True)

plt.axis("off")
plt.title("")
plt.imsave(f"{VIRUS}_unet_prediction.svg", np.squeeze(pred), vmin=-1, vmax=1)

plt.imsave(f"ground_truth.svg", np.squeeze(y), vmin=-1, vmax=1)
plt.imshow(np.squeeze(y), vmin=-1, vmax=1)
plt.axis("off")
plt.title("")
plt.close()

plt.imsave(f"input.svg", np.squeeze(x), vmin=-1, vmax=1)
plt.imshow(np.squeeze(x), vmin=-1, vmax=1)
plt.axis("off")
plt.title("")
plt.close()
