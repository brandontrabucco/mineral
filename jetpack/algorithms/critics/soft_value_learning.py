"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.value_learning import ValueLearning


class SoftValueLearning(ValueLearning):

    def __init__(
        self,
        vf,
        target_policy,
        target_vf,
        gamma=1.0,
        monitor=None,
    ):
        ValueLearning.__init__(
            self,
            vf,
            target_vf,
            gamma=gamma,
            monitor=monitor,
        )
        self.target_policy = target_policy

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        next_actions = self.target_policy.get_deterministic_actions(
            next_observations
        )
        epsilon = tf.clip_by_value(
            self.sigma * tf.random.normal(
                next_actions.shape,
                dtype=tf.float32
            ),
            -self.clip_radius,
            self.clip_radius
        )
        noisy_next_actions = next_actions + epsilon
        next_target_log_probs = self.target_policy.get_log_probs(
            next_observations,
            noisy_next_actions
        )
        next_target_values = self.target_vf.get_qvalues(
            next_observations
        )
        target_values = rewards + (
            terminals * self.gamma * (
                next_target_values - next_target_log_probs
            )
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "next_target_values_mean",
                tf.reduce_mean(next_target_values)
            )
            self.monitor.record(
                "next_target_log_probs_mean",
                tf.reduce_mean(next_target_log_probs)
            )
            self.monitor.record(
                "targets_mean",
                tf.reduce_mean(target_values)
            )
        return target_values
