"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC
from mineral.distributions.gaussians.gaussian import Gaussian


class TanhGaussian(Gaussian, ABC):

    def get_log_probs(
        self,
        x,
        *inputs,
        **kwargs
    ):
        x = tf.clip_by_value(x, -0.999, 0.999)
        correction = -1.0 * tf.reduce_sum(tf.math.log(1.0 - tf.math.square(x)), axis=-1)
        return correction + Gaussian.get_log_probs(
            self, tf.math.atanh(x), *inputs, **kwargs)

    def sample(
        self,
        *inputs,
        **kwargs
    ):
        return tf.math.tanh(Gaussian.sample(self, *inputs, **kwargs))

    def sample_from_prior(
        self,
        shape,
        **kwargs
    ):
        return tf.math.tanh(Gaussian.sample_from_prior(self, shape, **kwargs))

    def get_expected_value(
        self,
        *inputs,
        **kwargs
    ):
        return tf.math.tanh(Gaussian.get_expected_value(
            self, *inputs, **kwargs))

    def get_expected_value_of_prior(
        self,
        shape,
        **kwargs
    ):
        return tf.math.tanh(Gaussian.get_expected_value_of_prior(self, shape, **kwargs))
