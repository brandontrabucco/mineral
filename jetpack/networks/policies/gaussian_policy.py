"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import numpy as np
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


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

    def get_probs(
        self,
        observations,
        actions
    ):
        return tf.exp(self.get_log_probs(
            observations,
            actions
        ))

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

    def fisher_vector_product(
        self,
        observations,
        y
    ):
        with tf.GradientTape(persistent=True) as tape_policy:
            mean, std = self.get_mean_std(observations)
            mean_v = tf.ones(tf.shape(mean))
            tape_policy.watch(mean_v)
            mean_g = tape_policy.gradient(
                mean,
                self.trainable_variables,
                output_gradients=mean_v
            )
            std_v = tf.ones(tf.shape(std))
            tape_policy.watch(std_v)
            std_g = tape_policy.gradient(
                std,
                self.trainable_variables,
                output_gradients=std_v
            )
        mean_jvp = tape_policy.gradient(
            mean_g,
            mean_v,
            output_gradients=y
        )
        mean_fvp = tape_policy.gradient(
            mean,
            self.trainable_variables,
            output_gradients=mean_jvp
        )
        std_jvp = 2.0 / tf.square(std) * tape_policy.gradient(
            std_g,
            std_v,
            output_gradients=y
        )
        std_fvp = tape_policy.gradient(
            std,
            self.trainable_variables,
            output_gradients=std_jvp
        )
        return [m + s for m, s in zip(mean_fvp, std_fvp)]
