"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.critic import Critic


class ValueLearning(Critic):

    def __init__(
        self,
        vf,
        target_vf,
        gamma=1.0,
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
        next_target_values = self.target_vf.get_values(
            observations[:, 1:, ...]
        )
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * next_target_values[:, :, 0]
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
        weights = tf.tile([[self.gamma]], [1, tf.shape(rewards)[1]])
        weights = tf.math.cumprod(weights, axis=1, exclusive=True)
        discount_target_values = tf.math.cumsum(rewards * weights, axis=1, reverse=True) / weights
        if self.monitor is not None:
            self.monitor.record(
                "discount_target_values_mean",
                tf.reduce_mean(discount_target_values)
            )
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
            values = terminals[:, :(-1)] * self.vf.get_values(
                observations[:, :(-1), ...]
            )[:, :, 0]
            bellman_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    bellman_target_values,
                    values
                )
            )
            discount_loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    discount_target_values,
                    values
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "values_mean",
                    tf.reduce_mean(values)
                )
                self.monitor.record(
                    "bellman_loss_vf",
                    bellman_loss_vf
                )
                self.monitor.record(
                    "discount_loss_vf",
                    discount_loss_vf
                )
            return (
                self.bellman_weight * bellman_loss_vf +
                self.discount_weight * discount_loss_vf
            )
        self.vf.minimize(
            loss_function,
            observations[:, :(-1), ...]
        )

    def soft_update(
        self
    ):
        self.target_vf.soft_update(
            self.vf.get_weights()
        )

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        values = terminals[:, :(-1)] * self.vf.get_values(
            observations[:, :(-1), ...]
        )
        next_values = terminals[:, 1:] * self.vf.get_values(
            observations[:, 1:, ...]
        )
        return (
            rewards + next_values - values
        )