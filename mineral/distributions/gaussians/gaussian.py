"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf
from abc import ABC
from mineral.distributions.distribution import Distribution


class Gaussian(Distribution, ABC):

    def __init__(
        self,
        std=1.0
    ):
        self.std = std

    def get_parameters(
        self,
        *inputs,
        **kwargs
    ):
        activations = self.get_activations(*inputs, **kwargs)
        if self.std is None:
            mean, log_variance = tf.split(activations, 2, axis=-1)
        else:
            mean = activations
            log_variance = 2.0 * tf.math.log(tf.fill(
                tf.shape(mean), self.std))
        return mean, log_variance

    def get_log_probs(
        self,
        x,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        return -0.5 * tf.reduce_sum(
            tf.math.square(x - mean) / tf.math.exp(
                log_variance) + log_variance, axis=-1)

    def sample(
        self,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        return mean + tf.math.exp(0.5 * log_variance) * tf.random.normal(
            tf.shape(mean), dtype=tf.float32)

    def sample_from_prior(
        self,
        shape,
        **kwargs
    ):
        return tf.random.normal(shape, dtype=tf.float32)

    def get_expected_value(
        self,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        return mean

    def get_expected_value_of_prior(
        self,
        shape,
        **kwargs
    ):
        return tf.zeros(shape, dtype=tf.float32)

    def get_kl_divergence(
        self,
        pi,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        if pi is not None and pi != "prior":
            other_mean, other_log_variance = pi.get_parameters(*inputs, **kwargs)
        else:
            other_mean = tf.zeros(tf.shape(mean))
            other_log_variance = tf.zeros(tf.shape(log_variance))
        return 0.5 * tf.reduce_mean(
            other_log_variance - log_variance +
            tf.math.exp(log_variance - other_log_variance) +
            tf.square(mean - other_mean) / tf.math.exp(other_log_variance), axis=-1) - 0.5

    def get_fisher_information(
        self,
        *inputs,
        **kwargs
    ):
        mean, log_variance = self.get_parameters(*inputs, **kwargs)
        inv_variance = 1.0 / tf.math.exp(log_variance)
        return [inv_variance, 0.5 * tf.math.square(inv_variance)]
