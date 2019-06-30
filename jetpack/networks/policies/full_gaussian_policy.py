"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.gaussian_policy import GaussianPolicy


class FullGaussianPolicy(GaussianPolicy):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        GaussianPolicy.__init__(self, *args, **kwargs)

    def get_mean_std(
        self,
        activations
    ):
        mean, std = tf.split(
            activations,
            2,
            axis=-1
        )
        return mean, tf.math.softplus(std)