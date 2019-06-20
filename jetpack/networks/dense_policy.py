"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


class DensePolicy(DenseMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        mean=0.0,
        stddev=1.0,
        lower_bound=(-2.0),
        upper_bound=2.0,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)
        self.mean = mean
        self.stddev = stddev
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_stochastic_actions(
        self,
        observations
    ):
        x = self(observations)
        return x + tf.clip_by_value(
            tf.random.normal(x.shape, dtype=x.dtype) * self.stddev + self.mean,
            self.lower_bound,
            self.upper_bound)

    def get_deterministic_actions(
        self,
        observations
    ):
        return self(observations)
