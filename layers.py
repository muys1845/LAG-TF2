from tensorflow import keras
import tensorflow as tf


def conv(filters, kernel_size=3, strides=1, activation=None, trainable=False):
    return keras.layers.Conv2D(filters, kernel_size, strides, padding='same', activation=activation,
                               kernel_initializer='he_normal', trainable=trainable)


class ResBlock(keras.layers.Layer):
    """
    simple ResBlock used in Generator
    """

    def __init__(self, filters, ave):
        super(ResBlock, self).__init__()
        self.c1 = conv(filters, activation='relu')
        self.c2 = conv(filters)
        self.ave = ave
        self.config = {'filters': filters, 'ave': ave}

    def call(self, inputs):
        dy = self.c1(inputs)
        return inputs + self.c2(dy) / self.ave

    def get_config(self):
        return self.config


def upscale(x, scale: int = 2):
    """
    Box upscaling (also called nearest neighbors).

    Args:
    x: 4D tensor in NHWC format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor up scaled by a factor n.
    """
    b, h, w = tf.shape(x)[:-1]
    c = x.shape[-1]
    x = tf.reshape(x, [b, h, 1, w, 1, c])
    x = tf.tile(x, [1, 1, scale, 1, scale, 1])
    x = tf.reshape(x, [b, h * scale, w * scale, c])
    return x


def downscale(x, scale: int = 2):
    """
    Box downscaling.

    Args:
    x: 4D tensor.
    n: integer scale.
    order: NCHW or NHWC.

    Returns:
    4D tensor down scaled by a factor n.
    """
    return tf.nn.avg_pool(x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')


def space_to_channels(x, n=2):
    """Reshape image tensor by moving space to channels.

    Args:
    x: 4D tensor in NHWC format.
    n: integer scale (must be a power of 2).

    Returns:
    Reshaped 4D tensor image of shape (N, C * n**2, H // n, W // n).
    """
    s, ts = x.shape, tf.shape(x)

    x = tf.reshape(x, [-1, ts[1] // n, n, ts[2] // n, n, s[3]])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, ts[1] // n, ts[2] // n, s[3] * (n ** 2)])
    return x


def remove_details2d(x, n=2):
    """Remove box details by upscaling a downscaled image.

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor image with removed details of size nxn.
    """
    if n == 1:
        return x
    return upscale(downscale(x, n), n)


def blend_resolution(lores, hires, alpha):
    """Blend two images.

    Args:
        lores: 4D tensor in NCHW, low resolution image.
        hires: 4D tensor in NCHW, high resolution image.
        alpha: scalar tensor in [0, 1], 0 produces the low resolution, 1 the high one.

    Returns:
        4D tensor in NCHW of blended images.
    """
    return lores + alpha * (hires - lores)
