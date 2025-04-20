import numpy as np
import tensorflow as tf


def random_crop(x, y, img_size):
    """Randomly crops two images (x and y) to the specified size.

    The crop is centered around a randomly selected point within valid bounds.
    Both images receive the same crop coordinates to maintain spatial correspondence.

    Args:
        x (numpy.ndarray): First input image to crop.
        y (numpy.ndarray): Second input image to crop (must be same spatial dimensions as x).
        img_size (int): Desired output size (width and height) of the square crop.

    Returns:
        tuple: A tuple containing:
            - x_cropped (numpy.ndarray): Cropped version of x
            - y_cropped (numpy.ndarray): Cropped version of y
    """
    center_x = np.random.randint(img_size // 2, x.shape[1] - img_size // 2)
    center_y = np.random.randint(img_size // 2, x.shape[0] - img_size // 2)

    x = x[
        center_y - img_size // 2 : center_y + img_size // 2,
        center_x - img_size // 2 : center_x + img_size // 2,
    ]
    y = y[
        center_y - img_size // 2 : center_y + img_size // 2,
        center_x - img_size // 2 : center_x + img_size // 2,
    ]
    return x, y


def resize(x, y, img_size):
    """Resizes two images (x and y) to the specified square size using nearest-neighbor interpolation.

    Args:
        x (tensorflow.Tensor): First input image to resize.
        y (tensorflow.Tensor): Second input image to resize.
        img_size (int): Target size for both width and height.

    Returns:
        tuple: A tuple containing:
            - x_resized (tensorflow.Tensor): Resized version of x
            - y_resized (tensorflow.Tensor): Resized version of y
    """
    x = tf.image.resize(
        x, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    y = tf.image.resize(
        y, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return x, y


@tf.function()
def random_jitter(x, y):
    """Applies random data augmentation jitter to a pair of images.

    The augmentation pipeline consists of:
    1. Resizing to 286x286
    2. Random cropping back to 256x256
    3. Random horizontal flipping (50% chance)
    4. Random vertical flipping (50% chance)

    Args:
        x (tensorflow.Tensor): First input image to augment.
        y (tensorflow.Tensor): Second input image to augment (will receive identical transforms).

    Returns:
        tuple: A tuple containing:
            - x_augmented (tensorflow.Tensor): Augmented version of x
            - y_augmented (tensorflow.Tensor): Augmented version of y
    """
    # Resizing to 286x286
    x, y = resize(x, y, 286)

    # Random cropping back to 256x256
    x, y = random_crop(x, y, 256)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    if tf.random.uniform(()) > 0.5:
        # Random vertical flip
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)

    return x, y