"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.network import Network


class DenseNetwork(Network):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        Network.__init__(self, **kwargs)
        self.dense_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes
        ]

    def call(
        self,
        *inputs
    ):
        activations = self.dense_layers[0](tf.concat(inputs, -1))
        for layer in self.dense_layers[1:]:
            activations = layer(tf.nn.relu(activations))
        return activations
