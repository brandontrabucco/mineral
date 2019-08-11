"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf
from abc import ABC
from mineral.distributions.distribution import Distribution


class GaussianDistribution(Distribution, ABC):

    def __init__(
        self,
        std=1.0
    ):
        self.std = std

    def get_parameters(
        self,
        *inputs
    ):
        activations = self.get_activations(*inputs)
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
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        return -0.5 * tf.reduce_sum(
            tf.math.square(x - mean) / tf.math.exp(
                log_variance) + log_variance, axis=-1)

    def sample(
        self,
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        return mean + tf.math.exp(0.5 * log_variance) * tf.random.normal(
            tf.shape(mean), dtype=tf.float32)

    def get_expected_value(
        self,
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        return mean

    def get_kl_divergence(
        self,
        pi,
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        if pi is not None and pi != "prior":
            other_mean, other_log_variance = pi.get_parameters(*inputs)
        else:
            other_mean = tf.zeros(tf.shape(mean))
            other_log_variance = tf.zeros(tf.shape(log_variance))
        return 0.5 * tf.reduce_mean(
            other_log_variance - log_variance +
            tf.math.exp(log_variance - other_log_variance) +
            tf.square(mean - other_mean) / tf.math.exp(other_log_variance), axis=-1) - 0.5

    def get_fisher_information(
        self,
        *inputs
    ):
        mean, log_variance = self.get_parameters(*inputs)
        inv_variance = 1.0 / tf.math.exp(log_variance)
        return [inv_variance, 0.5 * tf.math.square(inv_variance)]
