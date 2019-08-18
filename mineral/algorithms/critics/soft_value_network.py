"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.value_network import ValueNetwork
from mineral import discounted_sum


class SoftValueNetwork(ValueNetwork):

    def __init__(
        self,
        policy,
        vf,
        target_vf,
        alpha=1.0,
        **kwargs
    ):
        ValueNetwork.__init__(
            self,
            vf,
            target_vf,
            **kwargs)
        self.policy = policy
        self.alpha = alpha

    def bellman_target_values(
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
        next_target_values = self.target_vf.get_expected_value(
            observations[:, 1:, ...])
        target_values = rewards - self.alpha * sampled_log_probs * terminals[:, :(-1)] + (
            terminals[:, 1:] * self.gamma * (
                next_target_values[:, :, 0]))
        self.record(
            "value_bellman_target_mean",
            tf.reduce_mean(target_values))
        return target_values

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        sampled_actions = self.policy.sample(
            observations[:, :(-1), ...])
        sampled_log_probs = terminals[:, :(-1)] * self.policy.get_log_probs(
            sampled_actions,
            observations[:, :(-1), ...])
        discount_target_values = discounted_sum((
            rewards - self.alpha * sampled_log_probs), self.gamma)
        self.record(
            "value_discount_target_mean",
            tf.reduce_mean(discount_target_values))
        return discount_target_values

    def update_critic(
        self,
        observations,
        actions,
        rewards,
        terminals,
        bellman_target_values,
        discount_target_values
    ):
        ValueNetwork.update_critic(
            self,
            observations,
            actions,
            rewards,
            terminals,
            bellman_target_values,
            discount_target_values)
