"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.mlp import MLP


class DenseMLP(MLP):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        MLP.__init__(self, **kwargs)
        self.hidden_layers = [
            tf.keras.layers.Dense(size)
            for size in hidden_sizes
        ]

    def call(
        self,
        observations
    ):
        x = self.hidden_layers[0](observations)
        for layer in self.hidden_layers[1:]:
            x = layer(tf.nn.relu(x))
        return x
