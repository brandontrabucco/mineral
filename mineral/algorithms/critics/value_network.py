"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.critic import Critic
from mineral import discounted_sum


class ValueNetwork(Critic):

    def __init__(
        self,
        vf,
        target_vf,
        gamma=0.99,
        bellman_weight=1.0,
        discount_weight=1.0,
        **kwargs
    ):
        Critic.__init__(self, **kwargs)
        self.master_vf = vf
        self.worker_vf = vf.clone()
        self.master_target_vf = target_vf
        self.worker_target_vf = target_vf.clone()
        self.gamma = gamma
        self.bellman_weight = bellman_weight
        self.discount_weight = discount_weight

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_target_values = self.worker_target_vf.get_expected_value(
            observations[:, 1:, ...])
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * next_target_values[:, :, 0])
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
        discount_target_values = discounted_sum(rewards, self.gamma)
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
        self.master_vf.copy_to(self.worker_vf)
        self.master_target_vf.copy_to(self.worker_target_vf)

        def loss_function():
            values = terminals[:, :(-1)] * self.worker_vf.get_expected_value(
                observations[:, :(-1), ...])[:, :, 0]
            bellman_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    bellman_target_values,
                    values))
            discount_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    discount_target_values,
                    values))
            self.record(
                "values_mean",
                tf.reduce_mean(values))
            self.record(
                "value_bellman_loss",
                bellman_loss_vf)
            self.record(
                "value_discount_loss",
                discount_loss_vf)
            return (
                self.bellman_weight * bellman_loss_vf +
                self.discount_weight * discount_loss_vf)
        self.worker_vf.minimize(
            loss_function,
            observations[:, :(-1), ...])
        self.worker_vf.copy_to(self.master_vf)
        self.worker_target_vf.copy_to(self.master_target_vf)

    def soft_update(
        self
    ):
        self.worker_target_vf.soft_update(self.worker_vf.get_weights())

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        values = terminals[:, :(-1)] * self.master_vf.get_expected_value(
            observations[:, :(-1), ...])
        next_values = terminals[:, 1:] * self.master_vf.get_expected_value(
            observations[:, 1:, ...])
        return rewards + next_values - values
