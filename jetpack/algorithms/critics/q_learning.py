"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class QLearning(Critic):

    def __init__(
        self,
        qf,
        target_policy,
        target_qf,
        gamma=1.0,
        sigma=1.0,
        clip_radius=1.0,
        monitor=None,
    ):
        self.qf = qf
        self.target_policy = target_policy
        self.target_qf = target_qf
        self.gamma = gamma
        self.sigma = sigma
        self.clip_radius = clip_radius
        self.iteration = 0
        self.monitor = monitor

    def get_qvalues(
        self,
        observations,
        actions
    ):
        return self.qf.get_qvalues(
            observations,
            actions
        )[:, 0]

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        next_actions = self.target_policy.get_deterministic_actions(
            next_observations
        )
        epsilon = tf.clip_by_value(
            self.sigma * tf.random.normal(
                tf.shape(next_actions),
                dtype=tf.float32
            ),
            -self.clip_radius,
            self.clip_radius
        )
        noisy_next_actions = next_actions + epsilon
        next_target_qvalues = self.target_qf.get_qvalues(
            next_observations,
            noisy_next_actions
        )[:, 0]
        target_values = rewards + (
            terminals * self.gamma * next_target_qvalues
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "next_target_qvalues_mean",
                tf.reduce_mean(next_target_qvalues)
            )
            self.monitor.record(
                "targets_mean",
                tf.reduce_mean(target_values)
            )
        return target_values

    def update_qf(
        self,
        observations,
        actions,
        target_values
    ):
        def loss_function():
            qvalues = self.qf.get_qvalues(
                observations,
                actions
            )[:, 0]
            loss_qf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    qvalues
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_qf",
                    loss_qf
                )
                self.monitor.record(
                    "qvalues_mean",
                    tf.reduce_mean(qvalues)
                )
            return loss_qf
        self.qf.minimize(
            loss_function,
            observations,
            actions
        )

    def soft_update(
        self
    ):
        self.target_qf.soft_update(
            self.qf.get_weights()
        )

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        target_values = self.get_target_values(
            rewards,
            next_observations,
            terminals
        )
        self.update_qf(
            observations,
            actions,
            target_values
        )
        self.soft_update()

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        self.gradient_update(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )
        return self.get_qvalues(
            observations,
            actions
        )
