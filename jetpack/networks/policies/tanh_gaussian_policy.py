"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.functions.policy import Policy


class TanhGaussianPolicy(GaussianPolicy, Policy):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        GaussianPolicy.__init__(self, hidden_sizes, **kwargs)

    def get_stochastic_actions(
        self,
        observations
    ):
        return tf.math.tanh(
            GaussianPolicy.get_stochastic_actions(
                self,
                observations
            )
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        return tf.math.tanh(
            GaussianPolicy.get_deterministic_actions(
                self,
                observations
            )
        )

    def get_log_probs(
        self,
        observations,
        actions
    ):
        actions = tf.clip_by_value(actions, -0.999, 0.999)
        correction = tf.reduce_sum(
            tf.math.log(1.0 - tf.math.square(actions)),
            axis=-1
        )
        return -1.0 * (
            correction + GaussianPolicy.get_log_probs(
                self,
                observations,
                tf.math.atanh(actions)
            )
        )
