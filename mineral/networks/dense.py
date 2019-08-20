"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.networks.network import Network
from mineral.core.cloneable import Cloneable


class Dense(Network):

    def __init__(
        self,
        hidden_sizes,
        dropout_rate=0.1,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        Cloneable.__init__(
            self,
            hidden_sizes,
            **kwargs)
        self.dense_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes]
        self.dense_bn_layers = [
            tf.keras.layers.BatchNormalization()
            for _i in range(len(self.dense_layers) - 1)]

    def call(
        self,
        *args,
        **kwargs
    ):
        inputs = tf.concat(args, -1)
        batch_shape = tf.shape(inputs)[:-1]
        inputs = tf.reshape(inputs, [tf.reduce_prod(batch_shape), tf.shape(inputs)[-1]])
        activations = tf.nn.relu(self.dense_layers[0](inputs))
        for i in range(1, len(self.dense_layers)):
            activations = self.dense_bn_layers[i - 1](activations, training=(activations.shape[0] > 1))
            activations = self.dense_layers[i](tf.nn.relu(activations))
        return tf.reshape(activations, tf.concat([
            batch_shape, tf.shape(activations)[-1:]], 0))
