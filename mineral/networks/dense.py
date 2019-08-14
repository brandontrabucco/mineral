"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network


class Dense(Network):

    def __init__(
        self,
        hidden_sizes,
        dropout_rate=0.2,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        self.dense_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes]
        self.dense_batch_norm_layers = [
            tf.keras.layers.BatchNormalization()
            for i in range(len(self.dense_layers))]
        self.dense_dropout_layers = [
            tf.keras.layers.Dropout(dropout_rate)
            for i in range(len(self.dense_layers))]

    def call(
        self,
        *args,
        training=False,
        **kwargs
    ):
        inputs = tf.concat(args, -1)
        batch_shape = tf.shape(inputs)[:-1]
        inputs = tf.reshape(inputs, [
            tf.reduce_prod(batch_shape), tf.shape(inputs)[-1]])
        activations = self.dense_batch_norm_layers[0](inputs, training=training)
        activations = self.dense_dropout_layers[0](activations, training=training)
        activations = tf.nn.relu(self.dense_layers[0](activations))
        for i in range(1, len(self.dense_layers)):
            activations = self.dense_batch_norm_layers[i](activations, training=training)
            activations = self.dense_dropout_layers[i](activations, training=training)
            activations = self.dense_layers[i](activations)
            if i < len(self.dense_layers) - 1:
                activations = tf.nn.relu(activations)
        return tf.reshape(activations, tf.concat([
            batch_shape, tf.shape(activations)[-1:]], 0))
