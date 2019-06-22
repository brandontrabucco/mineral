"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


class TanhGaussianPolicy(DenseMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)

    def get_stochastic_actions(
        self,
        observations
    ):
        mean, std = tf.split(self(observations), 2, axis=-1)
        std = tf.math.softplus(std)
        return tf.math.tanh(
            mean + std * tf.random.normal(
                mean.shape,
                dtype=tf.float32
            )
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        mean, std = tf.split(self(observations), 2, axis=-1)
        return tf.math.tanh(mean)

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
        mean, std = tf.split(self(observations), 2, axis=-1)
        correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.math.square(actions)),
            axis=-1
        )
        return -1.0 * (
            tf.losses.mean_squared_error(
                actions / std,
                mean / std
            ) + correction)
