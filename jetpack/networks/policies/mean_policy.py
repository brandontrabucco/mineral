"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.functions.policy import Policy
from jetpack.networks.dense_mlp import DenseMLP


class MeanPolicy(GaussianPolicy, Policy):

    def __init__(
        self,
        hidden_sizes,
        sigma=1.0,
        **kwargs
    ):
        GaussianPolicy.__init__(self, hidden_sizes, **kwargs)
        self.sigma = sigma

    def get_mean_std(
        self,
        observations
    ):
        mean = self(observations)
        return mean, tf.fill(tf.shape(mean), self.sigma)

    def naturalize(
        self,
        observations,
        y,
        tolerance=1e-3,
        maximum_iterations=100
    ):
        return DenseMLP.naturalize(
            self,
            observations,
            y,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations
        )
