"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.q_network import QNetwork
from mineral import discounted_sum


class SoftQNetwork(QNetwork):

    def __init__(
        self,
        *args,
        alpha=1.0,
        **kwargs
    ):
        QNetwork.__init__(
            self,
            *args,
            **kwargs)
        self.alpha = alpha

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_actions = self.worker_policy.sample(
            observations[:, 1:, ...])
        next_log_probs = self.worker_policy.get_log_probs(
            next_actions,
            observations[:, 1:, ...])
        next_target_qvalues = self.worker_target_qf.get_expected_value(
            observations[:, 1:, ...],
            next_actions)
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * (
                next_target_qvalues[:, :, 0] - self.alpha * next_log_probs))
        self.record("q_bellman_target_mean", tf.reduce_mean(target_values))
        return target_values

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        sampled_actions = self.worker_policy.sample(
            observations[:, :(-1), ...])
        sampled_log_probs = terminals[:, :(-1)] * self.worker_policy.get_log_probs(
            sampled_actions,
            observations[:, :(-1), ...])
        discount_target_values = discounted_sum((
            rewards - self.alpha * sampled_log_probs), self.gamma)
        self.record("q_discount_target_mean", tf.reduce_mean(discount_target_values))
        return discount_target_values
