"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC
from mineral.distributions.gaussians.gaussian_distribution import GaussianDistribution


class TanhGaussianDistribution(GaussianDistribution, ABC):

    def get_log_probs(
        self,
        x,
        *inputs,
        **kwargs
    ):
        x = tf.clip_by_value(x, -0.999, 0.999)
        correction = -1.0 * tf.reduce_sum(tf.math.log(1.0 - tf.math.square(x)), axis=-1)
        return correction + GaussianDistribution.get_log_probs(
            self, tf.math.atanh(x), *inputs, **kwargs)

    def sample(
        self,
        *inputs,
        **kwargs
    ):
        return tf.math.tanh(GaussianDistribution.sample(self, *inputs, **kwargs))

    def get_expected_value(
        self,
        *inputs,
        **kwargs
    ):
        return tf.math.tanh(GaussianDistribution.get_expected_value(
            self, *inputs, **kwargs))
