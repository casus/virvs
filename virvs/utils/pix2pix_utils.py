import numpy as np
import tensorflow as tf


def random_crop(x, y, mask, img_size):
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
    mask = mask[
        center_y - img_size // 2 : center_y + img_size // 2,
        center_x - img_size // 2 : center_x + img_size // 2,
    ]
    return x, y, mask


def resize(x, y, mask, img_size):
    x = tf.image.resize(
        x, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    y = tf.image.resize(
        y, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    mask = tf.image.resize(
        mask, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return x, y, mask


@tf.function()
def random_jitter(x, y, mask):
    # Resizing to 286x286
    x, y, mask = resize(x, y, mask, 286)

    # Random cropping back to 256x256
    x, y, mask = random_crop(x, y, mask, 256)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        # Random
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
        mask = tf.image.flip_up_down(mask)

    return x, y, mask
