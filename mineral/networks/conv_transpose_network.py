"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class ConvTransposeNetwork(Network):

    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        stride_sizes,
        hidden_sizes,
        initial_image_shape,
        dropout_rate=0.2,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        self.dense_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes]
        self.conv_layers = [
            tf.keras.layers.Conv2DTranspose(filters, kernels,
                                            strides=strides, padding="same")
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
        self.initial_image_shape = initial_image_shape

    def call(
        self,
        *inputs,
        training=False,
        **kwargs
    ):
        proprioceptive_inputs = [x for x in inputs if 2 <= len(x.shape) < 4]
        proprioceptive_inputs = tf.concat(proprioceptive_inputs, -1)
        batch_shape = tf.shape(proprioceptive_inputs)[:-1]
        proprioceptive_inputs = tf.reshape(
            proprioceptive_inputs,
            tf.concat([[tf.reduce_prod(batch_shape)],
                       tf.shape(proprioceptive_inputs)[-1:]], 0))
        activations = self.dense_layers[0](proprioceptive_inputs)
        for layer in self.dense_layers[1:]:
            activations = layer(tf.nn.relu(activations))
        activations = tf.reshape(activations, [
            tf.shape(activations)[0], *self.initial_image_shape])
        image_inputs = tf.concat([activations] + [
            x for x in inputs if len(x.shape) >= 4], -1)
        activations = tf.reshape(
            image_inputs,
            tf.concat([[tf.reduce_prod(batch_shape)],
                       tf.shape(image_inputs)[-3:]], 0))
        for i in range(len(self.conv_layers)):
            activations = self.batch_norm_layers[i](activations)
            activations = self.dropout_layers[i](activations)
            activations = self.conv_layers[i](tf.nn.relu(activations))
        return tf.reshape(activations, tf.concat([
            batch_shape, tf.shape(activations)[-3:]], 0))
