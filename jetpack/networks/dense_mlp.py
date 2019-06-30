"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.mlp import MLP
from jetpack.fisher import inverse_fisher_vector_product


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
        data
    ):
        x = self.hidden_layers[0](data)
        for layer in self.hidden_layers[1:]:
            x = layer(tf.nn.relu(x))
        return x

    def naturalize(
        self,
        data,
        grad,
        tolerance=1e-3,
        maximum_iterations=100
    ):
        return inverse_fisher_vector_product(
            lambda: [self(data)[0]],
            lambda mean: [tf.ones(tf.shape(mean))],
            self.trainable_variables,
            grad,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations
        )
