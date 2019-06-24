"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class QB(Critic):

    def __init__(
        self,
        qf1,
        qf2,
        target_policy,
        target_qf1,
        target_qf2,
        gamma=1.0,
        clip_radius=1.0,
        sigma=1.0,
        monitor=None,
    ):
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.gamma = gamma
        self.clip_radius = clip_radius
        self.sigma = sigma
        self.iteration = 0
        self.monitor = monitor
        target_qf1.set_weights(qf1.get_weights())
        target_qf2.set_weights(qf2.get_weights())

    def get_target_values(
        self,
        rewards,
        next_observations
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
        next_target_qvalues1 = self.target_qf1.get_qvalues(
            next_observations,
            noisy_next_actions
        )
        next_target_qvalues2 = self.target_qf2.get_qvalues(
            next_observations,
            noisy_next_actions
        )
        minimum_qvalues = tf.minimum(
            next_target_qvalues1,
            next_target_qvalues2
        )
        if self.monitor is not None:
            self.monitor.record(
                "next_target_qvalues1_mean",
                tf.reduce_mean(next_target_qvalues1)
            )
            self.monitor.record(
                "next_target_qvalues2_mean",
                tf.reduce_mean(next_target_qvalues2)
            )
            self.monitor.record(
                "minimum_qvalues_mean",
                tf.reduce_mean(minimum_qvalues)
            )
        return rewards + (self.gamma * minimum_qvalues)

    def update_qf1(
        self,
        observations,
        actions,
        target_values
    ):
        with tf.GradientTape() as tape_qf1:
            qvalues1 = self.qf1.get_qvalues(
                observations,
                actions
            )
            loss_qf1 = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    qvalues1
                )
            )
            self.qf1.minimize(
                loss_qf1,
                tape_qf1
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_qf1",
                    loss_qf1
                )
                self.monitor.record(
                    "qvalues1_mean",
                    tf.reduce_mean(qvalues1)
                )
            return qvalues1

    def update_qf2(
        self,
        observations,
        actions,
        target_values
    ):
        with tf.GradientTape() as tape_qf2:
            qvalues2 = self.qf2.get_qvalues(
                observations,
                actions
            )
            loss_qf2 = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    qvalues2
                )
            )
            self.qf2.minimize(
                loss_qf2,
                tape_qf2
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_qf2",
                    loss_qf2
                )
                self.monitor.record(
                    "qvalues2_mean",
                    tf.reduce_mean(qvalues2)
                )
            return qvalues2

    def gradient_update(
            self,
            observations,
            actions,
            rewards,
            next_observations
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        target_values = self.get_target_values(
            rewards,
            next_observations
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "target_values_mean",
                tf.reduce_mean(target_values)
            )
        qvalues1 = self.update_qf1(
            observations,
            actions,
            target_values
        )
        qvalues2 = self.update_qf2(
            observations,
            actions,
            target_values
        )
        self.target_qf1.soft_update(
            self.qf1.get_weights()
        )
        self.target_qf2.soft_update(
            self.qf2.get_weights()
        )
        return qvalues1, qvalues2

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        next_observations
    ):
        qvalues1, qvalues2 = self.gradient_update(
            observations,
            actions,
            rewards,
            next_observations
        )
        return tf.reduce_mean(qvalues1, qvalues2)

