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
        self.vf = vf
        self.target_vf = target_vf
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
        next_target_values = self.target_vf.get_expected_value(
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
        def loss_function():
            values = terminals[:, :(-1)] * self.vf.get_expected_value(
                observations[:, :(-1), ...])[:, :, 0]
            bellman_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    tf.stop_gradient(bellman_target_values),
                    values))
            discount_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    tf.stop_gradient(discount_target_values),
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
        self.vf.minimize(
            loss_function,
            observations[:, :(-1), ...])

    def soft_update(
        self
    ):
        self.vf.soft_update(self.vf.get_weights())

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        values = terminals[:, :(-1)] * self.vf.get_expected_value(
            observations[:, :(-1), ...])[:, :, 0]
        next_values = terminals[:, 1:] * self.vf.get_expected_value(
            observations[:, 1:, ...])[:, :, 0]
        return rewards + next_values - values
