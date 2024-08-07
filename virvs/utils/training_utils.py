import tensorflow as tf

from virvs.data.npy_dataloader import NpyDataloader


def prepare_dataset(
    path,
    im_size,
    ch_in,
    ch_out,
    random_jitter=False,
):

    dataloader = NpyDataloader(
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
