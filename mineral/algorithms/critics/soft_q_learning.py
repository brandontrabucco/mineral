"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.q_learning import QLearning
from mineral import discounted_sum


class SoftQLearning(QLearning):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        QLearning.__init__(
            self,
            *args,
            **kwargs
        )

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_actions = self.policy.get_expected_value(
            observations[:, 1:, ...]
        )
        next_log_probs = self.policy.get_log_probs(
            next_actions,
            observations[:, 1:, ...]
        )
        epsilon = tf.clip_by_value(
            self.std * tf.random.normal(
                tf.shape(next_actions),
                dtype=tf.float32
            ), -self.clip_radius, self.clip_radius
        )
        noisy_next_actions = next_actions + epsilon
        next_target_qvalues = self.target_qf.get_expected_value(
            observations[:, 1:, ...],
            noisy_next_actions
        )
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * (
                next_target_qvalues[:, :, 0] - next_log_probs
            )
        )
        if self.monitor is not None:
            self.monitor.record(
                "bellman_target_values_mean",
                tf.reduce_mean(target_values)
            )
        return target_values

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        log_probs = terminals[:, :(-1)] * self.policy.get_log_probs(
            actions,
            observations[:, :(-1), ...]
        )
        discount_target_values = discounted_sum((rewards - log_probs), self.gamma)
        if self.monitor is not None:
            self.monitor.record(
                "discount_target_values_mean",
                tf.reduce_mean(discount_target_values)
            )
        return discount_target_values