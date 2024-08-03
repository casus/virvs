from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile as tif
from virvs.utils.pix2pix_utils import random_jitter


def crop(x, center_x, center_y, im_size):
    x = x[
        center_y - im_size // 2 : center_y + im_size // 2,
        center_x - im_size // 2 : center_x + im_size // 2,
    ]

    return x


def center_crop(x, crop_size):
    x_center = x.shape[1] // 2
    y_center = x.shape[0] // 2

    return x[
        y_center - crop_size // 2 : y_center + crop_size // 2,
        x_center - crop_size // 2 : x_center + crop_size // 2,
    ]


class NpyDataloader:
    def __init__(
        self,
        path,
        im_size,
        random_jitter,
        ch_in,
        crop_type="random",
    ):

        self._x = np.load(join(path, "x.npy"))
        self._y = np.load(join(path, "y.npy"))

        if len(ch_in) == 1:
            self._x = self._x[..., ch_in[0] : ch_in[0] + 1]

        print("Data shape:", self._x.shape)
        self._im_size = im_size
        self._random_jitter = random_jitter
        self._crop_type = crop_type

    def __len__(self):
        len = self._x.shape[0]
        assert isinstance(len, int)
        return len

    def __getitem__(self, idx):
        x, y = self._x[idx], self._y[idx]

        if x.shape[0] > self._im_size or x.shape[1] > self._im_size:
            if self._crop_type == "random":
                center_x = np.random.randint(
                    self._im_size // 2, x.shape[1] - self._im_size // 2
                )
                center_y = np.random.randint(
                    self._im_size // 2, x.shape[0] - self._im_size // 2
                )

                x = crop(x, center_x, center_y, self._im_size)
                y = crop(y, center_x, center_y, self._im_size)
            elif self._crop_type == "center":
                x = center_crop(x, self._im_size)
                y = center_crop(y, self._im_size)
            else:
                assert False

        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)

        if self._random_jitter:
            x, y = random_jitter(x, y)

        return x, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
