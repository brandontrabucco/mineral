"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network
from mineral.core.cloneable import Cloneable


class Conv(Network, Cloneable):

    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        stride_sizes,
        hidden_sizes,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        Cloneable.__init__(
            self,
            filter_sizes,
            kernel_sizes,
            stride_sizes,
            hidden_sizes,
            **kwargs)
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
        *inputs,
        training=False,
        **kwargs
    ):
        image_inputs = tf.concat([x for x in inputs if len(x.shape) >= 4], -1)
        batch_shape = tf.shape(image_inputs)[:-3]
        conv_inputs = tf.reshape(
            image_inputs,
            tf.concat([[tf.reduce_prod(batch_shape)], tf.shape(image_inputs)[-3:]], 0))
        activations = tf.nn.relu(self.conv_layers[0](conv_inputs))
        for i in range(1, len(self.conv_layers)):
            activations = tf.nn.relu(self.conv_layers[i](activations))
        activations = tf.reshape(activations, tf.concat([batch_shape, [-1]], 0))
        proprioceptive_inputs = [x for x in inputs if 2 <= len(x.shape) < 4]
        activations = tf.concat([activations] + proprioceptive_inputs, -1)
        for i in range(len(self.dense_layers)):
            activations = self.dense_layers[i](activations)
            if i < len(self.dense_layers) - 1:
                activations = tf.nn.relu(activations)
        return activations
