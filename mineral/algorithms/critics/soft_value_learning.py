"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.value_learning import ValueLearning
from mineral import discounted_sum


class SoftValueLearning(ValueLearning):

    def __init__(
        self,
        policy,
        vf,
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
        self.policy = policy

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_actions = self.policy.get_deterministic_actions(
            observations[:, 1:, ...]
        )
        next_log_probs = self.policy.get_log_probs(
            next_actions,
            observations[:, 1:, ...]
        )
        next_target_values = self.target_vf.get_values(
            observations[:, 1:, ...]
        )
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * (
                next_target_values[:, :, 0] - next_log_probs
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
