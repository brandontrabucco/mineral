"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.mlp import MLP
from jetpack.core.policy import Policy
from jetpack.core.qf import QF


class FullyConnectedMLP(MLP):

    @staticmethod
    def flatten(
        x,
    ):
        return tf.reshape(
            x, 
            (x.shape[0], -1),
        )

    def __init__(
        self,
        hidden_sizes,
        **kwargs,
    ):
        MLP.__init__(self, **kwargs)
        self.layers = [
            tf.keras.layers.Dense(size) for size in hidden_sizes
        ]

    def __call__(
        self,
        observations,
    ):
        x = FullyConnectedMLP.flatten(observations)
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(tf.nn.relu(x))
        return x


class FullyConnectedPolicy(FullyConnectedMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        **kwargs,
    ):
        FullyConnectedMLP.__init__(self, hidden_sizes, **kwargs)

    def get_stochastic_actions(
        self,
        observations,
    ):
        x = self(observations)
        return x + tf.random.normal(x.shape)

    def get_deterministic_actions(
        self,
        observations,
    ):
        x = self(observations)
        return x


class FullyConnectedQF(FullyConnectedMLP, QF):

    def __init__(
        self,
        hidden_sizes,
        **kwargs,
    ):
        FullyConnectedMLP.__init__(self, hidden_sizes + [1], **kwargs)

    def get_qvalues(
        self,
        observations,
        actions,
    ):
        x = self(tf.concat([observations, actions], 1))
        return x