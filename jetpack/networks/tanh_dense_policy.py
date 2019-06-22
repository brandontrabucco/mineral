"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.policy import Policy


class TanhDensePolicy(DenseMLP, Policy):

    def __init__(
        self,
        hidden_sizes,
        sigma=1.0,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)
        self.sigma = sigma

    def get_stochastic_actions(
        self,
        observations
    ):
        mean = self(observations)
        return tf.math.tanh(
            mean + self.sigma * tf.random.normal(
                mean.shape,
                dtype=tf.float32
            )
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        return tf.math.tanh(self(observations))

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
        mean = self(observations)
        actions = tf.clip_by_value(actions, -0.999, 0.999)
        correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.math.square(actions)),
            axis=-1
        )
        return -1.0 * (tf.losses.mean_squared_error(
            tf.math.atanh(actions) / self.sigma,
            mean / self.sigma
        ) + correction)
