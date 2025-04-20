import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
    """
    Creates a downsampling block for a convolutional neural network.
    
    This function builds a block of layers consisting of:
    - A 2D convolutional layer with a specified number of filters and kernel size
    - Optionally, a batch normalization layer
    - A LeakyReLU activation function
    
    Args:
        filters (int): The number of filters for the convolutional layer.
        size (int): The size of the convolutional kernel (assumed to be square).
        apply_batchnorm (bool, optional): If True, applies batch normalization after the convolution. Default is True.
    
    Returns:
        tf.keras.Sequential: A TensorFlow Keras Sequential model containing the downsampling block.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    return result


def upsample(filters, size, apply_dropout=False):
    """
    Creates an upsampling block for a convolutional neural network.
    
    This function builds a block of layers consisting of:
    - A 2D transpose convolutional layer (also known as deconvolution)
    - A batch normalization layer
    - Optionally, a dropout layer
    - A ReLU activation function
    
    Args:
        filters (int): The number of filters for the transpose convolutional layer.
        size (int): The size of the transpose convolutional kernel (assumed to be square).
        apply_dropout (bool, optional): If True, applies dropout after batch normalization. Default is False.
    
    Returns:
        tf.keras.Sequential: A TensorFlow Keras Sequential model containing the upsampling block.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator(im_shape, ch_in, ch_out, apply_batchnorm=True, apply_dropout=True):
    """
    Builds a Generator model for a Pix2Pix-like architecture.
    
    The Generator model consists of an encoder-decoder (U-Net style):
    - A downsampling path that captures features at various levels.
    - An upsampling path that reconstructs the output image with skip connections from the encoder.
    
    Args:
        im_shape (int): The spatial size of the input image (assumed square).
        ch_in (int): The number of input channels (e.g., 3 for RGB images).
        ch_out (int): The number of output channels (e.g., 1 for grayscale images).
        apply_batchnorm (bool, optional): If True, applies batch normalization after convolution layers in the downsampling path. Default is True.
        apply_dropout (bool, optional): If True, applies dropout during the upsampling path to prevent overfitting. Default is True.
    
    Returns:
        tf.keras.Model: A Keras Model representing the Generator.
    """
    inputs = tf.keras.layers.Input(shape=[im_shape, im_shape, len(ch_in)])

    # Downsampling through the model
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 64, 64, 128)
        downsample(256, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 32, 32, 256)
        downsample(512, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 16, 16, 512)
        downsample(512, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 8, 8, 512)
        downsample(512, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 4, 4, 512)
        downsample(512, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 2, 2, 512)
        downsample(512, 4, apply_batchnorm=apply_batchnorm),  # (batch_size, 1, 1, 512)
    ]

    # Upsampling through the model
    up_stack = [
        upsample(512, 4, apply_dropout=apply_dropout),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=apply_dropout),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=apply_dropout),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    # Final convolution layer for generating the output image
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        ch_out,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )  # (batch_size, 256, 256, 1)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Skip connections from the encoder to the decoder (upsampling)
    skips = reversed(skips[:-1])  # Remove the last skip connection
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    # Final layer to generate the output
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=[x])


def Discriminator(im_shape, ch_in, ch_out):
    """
    Builds a Discriminator model for a Pix2Pix-like architecture.
    
    The Discriminator model classifies whether an image is real or generated by the Generator. 
    It consists of:
    - A series of downsampling convolutional layers
    - Final output layer that produces a single value indicating whether the input is real or fake.
    
    Args:
        im_shape (int): The spatial size of the input image (assumed square).
        ch_in (int): The number of input channels (e.g., 3 for RGB images).
        ch_out (int): The number of output channels (e.g., 1 for grayscale images).
    
    Returns:
        tf.keras.Model: A Keras Model representing the Discriminator.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    inp = tf.keras.layers.Input(shape=[im_shape, im_shape, len(ch_in)], name="input_image")
    tar = tf.keras.layers.Input(shape=[im_shape, im_shape, ch_out], name="target_image")

    # Concatenate the input and target images to make them both available for the discriminator
    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    # Downsampling path with multiple convolutional layers
    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    # Padding and convolution to reduce spatial dimensions further
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)  # (batch_size, 31, 31, 512)

    # Batch normalization and LeakyReLU activation
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.2)(batchnorm1)

    # Padding and final convolution to output a real/fake score
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
