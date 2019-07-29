"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class ConvNetwork(Network):

    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        stride_sizes,
        hidden_sizes,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters, kernels, strides=strides, padding="same")
            for filters, kernels, strides in zip(
                filter_sizes,
                kernel_sizes,
                stride_sizes)]
        self.dense_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes]

    def call(
        self,
        *inputs
    ):
        image_inputs = tf.concat([x for x in inputs if len(x.shape) >= 4], -1)
        batch_shape = tf.shape(image_inputs)[:-3]

        activations = self.conv_layers[0](tf.reshape(
            image_inputs,
            tf.concat([[tf.reduce_prod(batch_shape)], tf.shape(image_inputs)[-3:]], 0)))
        for layer in self.conv_layers[1:]:
            activations = layer(tf.nn.relu(activations))

        activations = tf.reshape(activations, tf.concat([batch_shape, [-1]], 0))
        proprioceptive_inputs = [x for x in inputs if 2 <= len(x.shape) < 4]

        activations = self.dense_layers[0](tf.concat([activations] + proprioceptive_inputs, -1))
        for layer in self.dense_layers[1:]:
            activations = layer(tf.nn.relu(activations))

        return activations
