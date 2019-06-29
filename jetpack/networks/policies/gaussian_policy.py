"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy
from jetpack.fisher import inverse_fisher_vector_product


class GaussianPolicy(DenseMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)

    def get_mean_std(
        self,
        observations
    ):
        mean, std = tf.split(self(observations), 2, axis=-1)
        return mean, tf.math.softplus(std)

    def get_stochastic_actions(
        self,
        observations
    ):
        mean, std = self.get_mean_std(observations)
        return mean + std * tf.random.normal(
            tf.shape(mean),
            dtype=tf.float32
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        return self.get_mean_std(observations)[0]

    def get_log_probs(
        self,
        observations,
        actions
    ):
        mean, std = self.get_mean_std(observations)
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
        mean, std = self.get_mean_std(observations)
        other_mean, other_std = other_policy.get_mean_std(observations)
        std_ratio = tf.square(std / other_std)
        return 0.5 * tf.reduce_sum(
            std_ratio +
            tf.square((other_mean - mean) / other_std) -
            tf.math.log(std_ratio) -
            tf.ones(tf.shape(mean)),
            axis=-1
        )

    def naturalize(
        self,
        observations,
        y,
        tolerance=1e-3,
        maximum_iterations=100
    ):
        return inverse_fisher_vector_product(
            lambda: self.get_mean_std(observations),
            lambda mean, std: [tf.ones(tf.shape(mean)), 2.0 / tf.square(std)],
            self.trainable_variables,
            y,
            tolerance=tolerance,
            maximum_iterations=maximum_iterations
        )
