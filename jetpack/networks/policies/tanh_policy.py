"""Author: Brandon Trabucco, Copyright 2019"""
from abc import ABC

import tensorflow as tf
from jetpack.functions.policy import Policy


class TanhPolicy(Policy):

    def __init__(
        self,
        policy
    ):
        self.policy = policy

    def get_stochastic_actions(
        self,
        observations
    ):
        return tf.math.tanh(
            self.policy.get_stochastic_actions(
                observations
            )
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        return tf.math.tanh(
            self.policy.get_deterministic_actions(
                observations
            )
        )

    def get_log_probs(
        self,
        observations,
        actions
    ):
        actions = tf.clip_by_value(actions, -0.999, 0.999)
        correction = -1.0 * tf.reduce_sum(
            tf.math.log(1.0 - tf.math.square(actions)),
            axis=-1
        )
        return correction + self.policy.get_log_probs(
            observations,
            tf.math.atanh(actions)
        )

    def get_kl_divergence(
        self,
        other_policy,
        observations
    ):
        return self.policy.get_kl_divergence(
            other_policy,
            observations
        )

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.policy, attr)
