"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


class DensePolicy(DenseMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        sigma=1.0,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)
        self.sigma = sigma

    def get_stochastic_actions(
        self,
        observations
    ):
        x = self(observations)
        return x + tf.random.normal(x.shape, dtype=x.dtype) * self.sigma

    def get_deterministic_actions(
        self,
        observations
    ):
        return self(observations)
