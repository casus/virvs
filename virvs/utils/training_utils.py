import tensorflow as tf

from virvs.data.npy_dataloader import NpyDataloader


def prepare_dataset(
    path,
    im_size,
    ch_in,
    ch_out,
    random_jitter=False,
):
    """Prepares a TensorFlow dataset from NPY files using a custom dataloader.

    Creates a tf.data.Dataset pipeline that loads and optionally augments image data
    from numpy files. The dataset yields pairs of input and output images with
    specified channel configurations.

    Args:
        path (str): Path to the directory containing the numpy files.
        im_size (int): Target size for the images (width and height).
        ch_in (list): List of channel indices to use for input images.
        ch_out (int): Number of channels in output images.
        random_jitter (bool, optional): Whether to apply random data augmentation.
            Defaults to False.

    Returns:
        tf.data.Dataset: A TensorFlow dataset yielding tuples of:
            - input_image (tf.Tensor): Input image tensor with shape 
              [im_size, im_size, len(ch_in)] and dtype tf.float32
            - output_image (tf.Tensor): Output image tensor with shape
              [im_size, im_size, ch_out] and dtype tf.float32
    """
    dataloader=NpyDataloader(
        path=path,
        im_size=im_size,
        random_jitter=random_jitter,
        ch_in=ch_in,
        crop_type="random",
    )
    dataset = tf.data.Dataset.from_generator(
        dataloader,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [im_size, im_size, len(ch_in)],
            [im_size, im_size, ch_out],
        ),
    )
    return dataset
