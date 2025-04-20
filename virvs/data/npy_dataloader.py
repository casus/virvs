from os import listdir
from os.path import isfile, join

import numpy as np
import tifffile as tif
from virvs.utils.pix2pix_utils import random_jitter


def crop(x, center_x, center_y, im_size):
    """
    Crop the input image `x` to the specified size around a given center.

    This function takes an image and crops it around a specified center (center_x, center_y)
    to the size defined by `im_size`.

    Args:
        x (np.ndarray): The input image to crop.
        center_x (int): The x-coordinate of the center of the crop.
        center_y (int): The y-coordinate of the center of the crop.
        im_size (int): The size of the output cropped image (width and height).

    Returns:
        np.ndarray: The cropped image of size (im_size, im_size).
    """
    x = x[
        center_y - im_size // 2 : center_y + im_size // 2,
        center_x - im_size // 2 : center_x + im_size // 2,
    ]
    return x


def center_crop(x, crop_size):
    """
    Crop the input image `x` to the specified crop size from its center.

    This function crops the input image around its center to a square of size `crop_size`.

    Args:
        x (np.ndarray): The input image to crop.
        crop_size (int): The size of the square crop.

    Returns:
        np.ndarray: The cropped image of size (crop_size, crop_size).
    """
    x_center = x.shape[1] // 2
    y_center = x.shape[0] // 2

    return x[
        y_center - crop_size // 2 : y_center + crop_size // 2,
        x_center - crop_size // 2 : x_center + crop_size // 2,
    ]


class NpyDataloader:
    """
    A data loader class for loading and preprocessing `.npy` image data.

    This class is designed to load image pairs from `.npy` files stored at the 
    specified path, with support for random or center cropping, and random jittering.
    
    Attributes:
        _x (np.ndarray): The input images (features).
        _y (np.ndarray): The target images (labels).
        _im_size (int): The size of the images to which they should be cropped.
        _random_jitter (bool): Whether to apply random jittering.
        _crop_type (str): The type of cropping ("random" or "center").
    """

    def __init__(
        self,
        path,
        im_size,
        random_jitter,
        ch_in,
        crop_type="random",
    ):
        """
        Initialize the NpyDataloader.

        This constructor loads the `.npy` files from the given path, slices the input 
        images according to the `ch_in` channels, and sets up the other configurations 
        (image size, cropping, jittering).

        Args:
            path (str): The directory containing the 'x.npy' and 'y.npy' files.
            im_size (int): The size to which the images should be cropped.
            random_jitter (bool): Whether to apply random jittering to the images.
            ch_in (List[int]): The list of input channels to be selected from the images.
            crop_type (str): The type of cropping ('random' or 'center'). Default is 'random'.
        """
        self._x = np.load(join(path, "x.npy"))
        self._y = np.load(join(path, "y.npy"))

        if len(ch_in) == 1:
            self._x = self._x[..., ch_in[0] : ch_in[0] + 1]

        print("Data shape:", self._x.shape)
        self._im_size = im_size
        self._random_jitter = random_jitter
        self._crop_type = crop_type

    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        length = self._x.shape[0]
        assert isinstance(length, int)
        return length

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        This function returns the input image `x` and target image `y` at the specified 
        index `idx`, after applying cropping and optional random jittering.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the input image `x` and the target image `y`.
        """
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
        """
        Yield samples one by one.

        This function allows the dataloader to be used in a loop, yielding one sample 
        (input image and target image) at a time.

        Yields:
            tuple: A tuple containing the input image `x` and the target image `y`.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)
