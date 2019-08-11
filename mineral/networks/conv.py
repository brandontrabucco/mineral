"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class Conv(Network):

    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        stride_sizes,
        hidden_sizes,
        dropout_rate=0.2,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters, kernels, strides=strides, padding="same")
            for filters, kernels, strides in zip(
                filter_sizes,
                kernel_sizes,
                stride_sizes)]
        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for i in range(len(self.conv_layers))]
        self.dropout_layers = [
            tf.keras.layers.SpatialDropout2D(dropout_rate)
            for i in range(len(self.conv_layers))]
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
        activations = self.conv_layers[0](conv_inputs)
        activations = self.batch_norm_layers[0](activations, training=training)
        activations = self.dropout_layers[0](activations, training=training)
        for i in range(1, len(self.conv_layers)):
            activations = self.conv_layers[i](tf.nn.relu(activations))
            activations = self.batch_norm_layers[i](activations, training=training)
            activations = self.dropout_layers[i](activations, training=training)
        activations = tf.reshape(activations, tf.concat([batch_shape, [-1]], 0))
        proprioceptive_inputs = [x for x in inputs if 2 <= len(x.shape) < 4]
        activations = tf.concat([activations] + proprioceptive_inputs, -1)
        for layer in self.dense_layers:
            activations = layer(tf.nn.relu(activations))
        return activations
