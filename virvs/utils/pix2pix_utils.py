import numpy as np
import tensorflow as tf


def random_crop(x, y, img_size):
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
    x = tf.image.resize(
        x, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    y = tf.image.resize(
        y, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return x, y


@tf.function()
def random_jitter(x, y):
    # Resizing to 286x286
    x, y = resize(x, y, 286)

    # Random cropping back to 256x256
    x, y = random_crop(x, y, 256)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)

    if tf.random.uniform(()) > 0.5:
        # Random
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)

    return x, y
