from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile as tif
from tqdm import tqdm

PATH = "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed"

for split in tqdm(["train", "val", "test"]):
    path = join(PATH, split)
    # load the images
    x = []
    y = []
    masks = []
    filenames = [f for f in listdir(join(path, "x")) if isfile(join(path, "x", f))]
    for f in tqdm(filenames):
        x.append(tif.imread(join(path, "x", f)))
        y.append(tif.imread(join(path, "gt", f)))
        # for HAdV uncomment:
        # masks.append(np.load(join(path, "masks", f.replace(".tif", ".npy"))))
    np.save(join(path, f"x.npy"), np.array(x))
    np.save(join(path, f"y.npy"), np.array(y))
    # for HAdV uncomment:
    # np.save(join(path, f"masks.npy"), np.array(masks))
