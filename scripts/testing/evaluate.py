from collections import defaultdict

import numpy as np
from tqdm import tqdm
from virvs.data.npy_dataloader import NpyDataloader
from virvs.utils.evaluation_utils import evaluate

outputs = np.load(f"")
virus = "vacv"

if "vacv" not in virus:
    size = 2048
else:
    size = 5888
# crop center if not fitting
dataloader = NpyDataloader(
    path="",
    im_size=size,
    random_jitter=False,
    crop_type="center",
    ch_in=[0],
)

evaluate(outputs, dataloader._y)
