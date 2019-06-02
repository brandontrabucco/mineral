"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import jetpack as jp
from jetpack.networks.mlp import MLP
from jetpack.core.policy import Policy
from jetpack.core.qf import QF


class FullyConnectedMLP(MLP):

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
        x = self(jp.flatten(observations))
        return x + tf.random.normal(x.shape)

    def get_deterministic_actions(
        self,
        observations,
    ):
        return self(jp.flatten(observations))


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
        return self(tf.concat([
            jp.flatten(observations), 
            jp.flatten(actions),
        ], 1))