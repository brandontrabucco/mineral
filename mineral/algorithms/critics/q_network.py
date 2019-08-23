"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.critic import Critic
from mineral import discounted_sum


class QNetwork(Critic):

    def __init__(
        self,
        policy,
        qf,
        target_qf,
        gamma=0.99,
        bellman_weight=1.0,
        discount_weight=1.0,
        **kwargs
    ):
        Critic.__init__(self, **kwargs)
        self.policy = policy
        self.qf = qf
        self.target_qf = target_qf
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
        next_actions = self.policy.get_expected_value(
            observations[:, 1:, ...])
        next_target_qvalues = self.target_qf.get_expected_value(
            observations[:, 1:, ...],
            next_actions)
        target_values = rewards + (
            terminals[:, 1:] * self.gamma * next_target_qvalues[:, :, 0])
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
        discount_target_values = discounted_sum(rewards, self.gamma)
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
                actions)[:, :, 0]
            bellman_loss_qf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    tf.stop_gradient(bellman_target_values),
                    qvalues))
            discount_loss_qf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    tf.stop_gradient(discount_target_values),
                    qvalues))
            self.record(
                "qvalues_mean",
                tf.reduce_mean(qvalues))
            self.record(
                "qvalues_max",
                tf.reduce_max(qvalues))
            self.record(
                "qvalues_min",
                tf.reduce_min(qvalues))
            self.record(
                "q_bellman_loss",
                bellman_loss_qf)
            self.record(
                "q_discount_loss",
                discount_loss_qf)
            return (self.bellman_weight * bellman_loss_qf +
                    self.discount_weight * discount_loss_qf)
        self.qf.minimize(
            loss_function,
            observations[:, :(-1), ...],
            actions)

    def soft_update(
        self
    ):
        self.target_qf.soft_update(self.qf.get_weights())

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals,
    ):
        return terminals[:, :(-1)] * self.qf.get_expected_value(
            observations[:, :(-1), ...], actions)[:, :, 0]
