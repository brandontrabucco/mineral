"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from abc import ABC
from jetpack.distributions.gaussian_distribution import GaussianDistribution


class TanhGaussianDistribution(GaussianDistribution, ABC):

    def get_log_probs(
        self,
        x,
        *inputs
    ):
        x = tf.clip_by_value(x, -0.999, 0.999)
        correction = -1.0 * tf.reduce_sum(tf.math.log(1.0 - tf.math.square(x)), axis=-1)
        return correction + self.policy.get_log_probs(
            tf.math.atanh(x), *inputs)

    def sample(
        self,
        *inputs
    ):
        return tf.math.tanh(self.distribution.sample(*inputs))

    def get_expected_value(
        self,
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        return tf.tanh(mean)
