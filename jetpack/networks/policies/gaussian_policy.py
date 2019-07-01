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

    def get_mean_std(
        self,
        activations
    ):
        if self.std is None:
            mean, std = tf.split(
                activations,
                2,
                axis=-1
            )
            std = tf.math.softplus(std)
        else:
            mean = activations
            std = tf.fill(
                tf.shape(mean),
                self.std
            )
        return mean, std

    def call(
        self,
        observations
    ):
        return self.get_mean_std(
            DenseMLP.__call__(self, observations)
        )

    def get_stochastic_actions(
        self,
        observations
    ):
        mean, std = self(observations)
        return mean + std * tf.random.normal(
            tf.shape(mean),
            dtype=tf.float32
        )

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
        mean, std = self(observations)
        return -0.5 * tf.reduce_sum(
            tf.math.square((actions - mean) / std) + tf.math.log(
                tf.math.square(std) * tf.fill(tf.shape(mean), 2.0 * np.pi)
            ),
            axis=-1
        )

    def get_kl_divergence(
        self,
        other_policy,
        observations
    ):
        mean, std = self(observations)
        other_mean, other_std = other_policy(observations)
        std_ratio = tf.square(std / other_std)
        return 0.5 * tf.reduce_sum(
            std_ratio +
            tf.square((other_mean - mean) / other_std) -
            tf.math.log(std_ratio) -
            tf.ones(tf.shape(mean)),
            axis=-1
        )
