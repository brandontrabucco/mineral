"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.policy_gradient import PolicyGradient


class NaturalPolicyGradient(PolicyGradient):

    def __init__(
        self,
        policy,
        gamma=1.0,
        monitor=None,
    ):
        PolicyGradient.__init__(
            self,
            policy,
            gamma=gamma,
            monitor=monitor,
        )

    def update_policy(
        self,
        observations,
        actions,
        returns
    ):
        with tf.GradientTape() as tape_policy:
            loss_policy = -1.0 * tf.reduce_mean(
                returns * self.policy.get_log_probs(
                    observations[:, :(-1), :],
                    actions
                )
            )
            grad = tape_policy.gradient(
                loss_policy,
                self.policy.trainable_variables
            )
            grad, sAs = self.policy.naturalize(
                observations,
                grad
            )
            self.policy.apply_gradients(
                grad
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    tf.reduce_mean(loss_policy)
                )
