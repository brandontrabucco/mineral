"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC
from mineral.distributions.gaussians.gaussian_distribution import GaussianDistribution


class ExpGaussianDistribution(GaussianDistribution, ABC):

    def get_log_probs(
        self,
        x,
        *inputs,
        **kwargs
    ):
        x = tf.maximum(x, 0.001)
        correction = -1.0 * tf.reduce_sum(x, axis=-1)
        return correction + GaussianDistribution.get_log_probs(
            self, tf.math.log(x), *inputs, **kwargs)

    def sample(
        self,
        *inputs,
        **kwargs
    ):
        return tf.math.exp(GaussianDistribution.sample(self, *inputs, **kwargs))

    def get_expected_value(
        self,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        return tf.exp(mean + 0.5 * tf.math.exp(log_variance))
