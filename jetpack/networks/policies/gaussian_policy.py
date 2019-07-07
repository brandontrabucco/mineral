"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf
from jetpack.networks.dense.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


class GaussianPolicy(DenseMLP, Policy):

    def __init__(
        self,
        *args,
        std=None,
        **kwargs
    ):
        DenseMLP.__init__(self, *args, **kwargs)
        self.std = std

    def get_mean_log_variance(
        self,
        activations
    ):
        if self.std is None:
            mean, log_variance = tf.split(
                activations, 2, axis=-1)
        else:
            mean = activations
            log_variance = 2.0 * tf.math.log(
                tf.fill(tf.shape(mean), self.std))
        return mean, log_variance

    def call(
        self,
        observations
    ):
        return self.get_mean_log_variance(
            DenseMLP.call(self, observations)
        )

    def get_stochastic_actions(
        self,
        observations
    ):
        mean, log_variance = self(observations)
        return mean + tf.math.exp(0.5 * log_variance) * tf.random.normal(
            tf.shape(mean), dtype=tf.float32)

    def get_deterministic_actions(
        self,
        observations
    ):
        return self(observations)[0]

    def get_log_probs(
        self,
        observations,
        actions
    ):
        mean, log_variance = self(observations)
        return -0.5 * tf.reduce_sum(
            tf.math.square(actions - mean) / tf.math.exp(
                log_variance) + log_variance + tf.math.log(
                    tf.fill(tf.shape(mean), 2.0 * np.pi)), axis=-1)

    def get_kl_divergence(
        self,
        other_policy,
        observations
    ):
        mean, log_variance = self(observations)
        other_mean, other_log_variance = other_policy(observations)
        return 0.5 * tf.reduce_sum(
            tf.math.exp(log_variance - other_log_variance) +
            other_log_variance - log_variance -
            tf.square(other_mean - mean) / tf.math.exp(
                other_log_variance) +
            tf.ones(tf.shape(mean)), axis=-1)
