"""Author: Brandon Trabucco, Copyright 2019"""

import tensorflow as tf
from jetpack.functions.policy import Policy


class SoftplusPolicy(Policy):

    def __init__(
        self,
        policy
    ):
        self.policy = policy

    def get_stochastic_actions(
        self,
        observations
    ):
        return tf.math.softplus(
            self.policy.get_stochastic_actions(
                observations
            )
        )

    def get_deterministic_actions(
        self,
        observations
    ):
        return tf.math.softplus(
            self.policy.get_deterministic_actions(
                observations
            )
        )

    def get_log_probs(
        self,
        observations,
        actions
    ):
        actions = tf.maximum(actions, 0.001)
        correction = -1.0 * tf.reduce_sum(
            tf.math.log_sigmoid(actions),
            axis=-1
        )
        return correction + self.policy.get_log_probs(
            observations,
            tf.math.log(tf.math.exp(actions) - 1)
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

    def __call__(
        self,
        observations
    ):
        return self.policy(observations)

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.policy, attr)
