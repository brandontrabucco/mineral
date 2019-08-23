"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.q_network import QNetwork
from mineral import discounted_sum


class SoftQNetwork(QNetwork):

    def __init__(
        self,
        *args,
        log_alpha=0.0,
        **kwargs
    ):
        QNetwork.__init__(
            self,
            *args,
            **kwargs)
        self.log_alpha = log_alpha

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_actions = self.policy.get_expected_value(
            observations[:, 1:, ...])
        next_log_probs = self.policy.get_log_probs(
            next_actions,
            observations[:, 1:, ...])
        next_target_qvalues = self.target_qf.get_expected_value(
            observations[:, 1:, ...],
            next_actions)
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * (
                next_target_qvalues[:, :, 0] - tf.exp(self.log_alpha) * next_log_probs))
        self.record(
            "q_bellman_target_mean",
            tf.reduce_mean(target_values))
        self.record(
            "q_bellman_target_max",
            tf.reduce_max(target_values))
        self.record(
            "q_bellman_target_min",
            tf.reduce_min(target_values))
        return target_values

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        sampled_actions = self.policy.get_expected_value(
            observations[:, :(-1), ...])
        sampled_log_probs = terminals[:, :(-1)] * self.policy.get_log_probs(
            sampled_actions,
            observations[:, :(-1), ...])
        discount_target_values = discounted_sum((
                rewards - tf.exp(self.log_alpha) * sampled_log_probs), self.gamma)
        self.record(
            "q_discount_target_mean",
            tf.reduce_mean(discount_target_values))
        self.record(
            "q_discount_target_max",
            tf.reduce_max(discount_target_values))
        self.record(
            "q_discount_target_min",
            tf.reduce_min(discount_target_values))
        return discount_target_values
