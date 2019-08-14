"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.critic import Critic
from mineral import discounted_sum


class QLearning(Critic):

    def __init__(
        self,
        policy,
        qf,
        target_qf,
        gamma=1.0,
        std=1.0,
        clip_radius=1.0,
        bellman_weight=1.0,
        discount_weight=1.0,
        **kwargs
    ):
        Critic.__init__(self, **kwargs)
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
        self.gamma = gamma
        self.std = std
        self.clip_radius = clip_radius
        self.bellman_weight = bellman_weight
        self.discount_weight = discount_weight

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        next_actions = self.policy.get_expected_value(
            observations[:, 1:, ...])
        epsilon = tf.clip_by_value(
            self.std * tf.random.normal(
                tf.shape(next_actions),
                dtype=tf.float32), -self.clip_radius, self.clip_radius)
        noisy_next_actions = next_actions + epsilon
        next_target_qvalues = self.target_qf.get_expected_value(
            observations[:, 1:, ...],
            noisy_next_actions)
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * next_target_qvalues[:, :, 0])
        if self.monitor is not None:
            self.monitor.record(
                "bellman_target_values_mean",
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
        if self.monitor is not None:
            self.monitor.record(
                "discount_target_values_mean",
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
            qvalues = terminals[:, :(-1)] * self.qf.get_expected_value(
                observations[:, :(-1), ...],
                actions,
                training=True)[:, :, 0]
            bellman_loss_qf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    bellman_target_values,
                    qvalues))
            discount_loss_qf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    discount_target_values,
                    qvalues))
            if self.monitor is not None:
                self.monitor.record(
                    "qvalues_mean",
                    tf.reduce_mean(qvalues))
                self.monitor.record(
                    "bellman_loss_qf",
                    bellman_loss_qf)
                self.monitor.record(
                    "discount_loss_qf",
                    discount_loss_qf)
            return (
                self.bellman_weight * bellman_loss_qf +
                self.discount_weight * discount_loss_qf)
        self.qf.minimize(
            loss_function,
            observations[:, :(-1), ...],
            actions)

    def soft_update(
        self
    ):
        self.target_qf.soft_update(
            self.qf.get_weights())

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals,
    ):
        qvalues = self.qf.get_expected_value(
            observations[:, :(-1), ...],
            actions)
        values = self.qf.get_expected_value(
            observations[:, :(-1), ...],
            self.policy.get_expected_value(
                observations[:, :(-1), ...]))
        return terminals[:, :(-1)] * (
            qvalues[:, :, 0] - values[:, :, 0])
