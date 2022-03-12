import tensorflow as tf
from tensorflow import keras
import layers
import numpy as np


class Generator(keras.Model):
    def __init__(self, lfilters, blocks, noise_dim):
        super(Generator, self).__init__()
        self.c1 = layers.conv(lfilters[0], activation='relu', trainable=True)
        self.block = keras.Sequential([layers.ResBlock(lfilters[0], blocks) for _ in range(blocks)])
        self.conv = [
            [layers.conv(filters, activation=tf.nn.leaky_relu) for _ in range(2)]
            for filters in lfilters[1:]
        ]
        self.to_rgb = [layers.conv(3) for _ in lfilters[1:]]
        self.noise_dim = noise_dim
        self.config = {'lfilters': lfilters, 'blocks': blocks, 'noise_dim': noise_dim}

    def update_trainable_layers(self, lod_stop):
        for stage in range(lod_stop):
            for layer in self.conv[stage]:
                layer.trainable = True
            self.to_rgb[stage].trainable = True

    def call(self, inputs, add_noise=False, lod=None, lod_start=None, lod_stop=None):
        if add_noise:
            noise = tf.random.normal([*inputs.shape[:-1], self.noise_dim])
        else:
            noise = tf.zeros([*inputs.shape[:-1], self.noise_dim])
        y = self.c1(tf.concat([inputs, noise], axis=3))
        y = self.block(y)
        rgb = []
        for stage in range(lod_stop):
            y = layers.upscale(y)
            y = self.conv[stage][0](y)
            y = self.conv[stage][1](y)
            rgb.append(self.to_rgb[stage](y))
        im = rgb.pop(0)
        for _ in range(1, lod_start):
            im = layers.upscale(im) + rgb.pop(0)
        if lod_start == lod_stop:
            return im
        return layers.upscale(im) + (lod - lod_start) * rgb[-1]

    def get_config(self):
        return self.config


class Discriminator(keras.Model):
    def __init__(self, lfilters, blocks):
        super(Discriminator, self).__init__()
        self.conv = [
            [layers.conv(lfilters[stage], activation=tf.nn.leaky_relu) for _ in range(2)] +
            [layers.conv(lfilters[stage - 1], activation=tf.nn.leaky_relu)]
            for stage in range(1, len(lfilters))
        ]
        self.block = keras.Sequential([
            layers.conv(lfilters[0], activation=tf.nn.leaky_relu, trainable=True)
            for _ in range(blocks)
        ])
        self.config = {'lfilters': lfilters, 'blocks': blocks}
        center = np.ones(lfilters[0], 'f')
        center[::2] = -1
        self.center = tf.constant(center, shape=[1, 1, 1, lfilters[0]])

    def update_trainable_layers(self, lod_stop):
        for stage in range(lod_stop):
            for layer in self.conv[stage]:
                layer.trainable = True

    def call(self, inputs, lores_delta=None, lod=None, lod_start=None, lod_stop=None):
        y = None
        for stage in range(lod_stop, 0, -1):
            if stage == lod_stop:
                y = self.conv[stage - 1][0](inputs)
            elif stage == lod_start:
                y0 = self.conv[stage - 1][0](layers.downscale(inputs))
                y = y0 + (lod - lod_start) * y
            else:
                y += self.conv[stage - 1][0](layers.downscale(inputs, 1 << (lod_stop - stage)))
            y = self.conv[stage - 1][1](y)
            y = layers.space_to_channels(y)
            y = self.conv[stage - 1][2](y)
        y = tf.concat([y, lores_delta], axis=3)
        y = self.block(y)
        return y * self.center

    def get_config(self):
        return self.config
