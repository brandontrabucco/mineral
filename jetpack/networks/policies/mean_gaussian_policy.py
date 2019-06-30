"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.gaussian_policy import GaussianPolicy


class MeanGaussianPolicy(GaussianPolicy):

    def __init__(
        self,
        *args,
        std=1.0,
        **kwargs
    ):
        GaussianPolicy.__init__(self, *args, **kwargs)
        self.std = std

    def get_mean_std(
        self,
        activations
    ):
        return activations, tf.fill(
            tf.shape(activations),
            self.std
        )