"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class ValueLearning(Critic):

    def __init__(
        self,
        vf,
        target_vf,
        gamma=1.0,
        monitor=None,
    ):
        self.vf = vf
        self.target_vf = target_vf
        self.gamma = gamma
        self.iteration = 0
        self.monitor = monitor

    def get_values(
        self,
        observations
    ):
        return self.vf.get_values(
            observations
        )[:, 0]

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        next_target_values = self.target_vf.get_qvalues(
            next_observations
        )[:, 0]
        target_values = rewards + (
            terminals * self.gamma * next_target_values
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "next_target_values_mean",
                tf.reduce_mean(next_target_values)
            )
            self.monitor.record(
                "targets_mean",
                tf.reduce_mean(target_values)
            )
        return target_values

    def update_vf(
        self,
        observations,
        target_values,
        terminals
    ):
        def loss_function():
            values = self.vf.get_values(
                observations,
                terminals
            )[:, 0]
            loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    values
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_vf",
                    loss_vf
                )
                self.monitor.record(
                    "values_mean",
                    tf.reduce_mean(values)
                )
            return loss_vf
        self.vf.minimize(
            loss_function
        )

    def soft_update(
        self
    ):
        self.target_vf.soft_update(
            self.vf.get_weights()
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
        self.update_vf(
            observations,
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
        next_values = self.get_values(
            next_observations
        )
        if self.monitor is not None:
            self.monitor.record(
                "next_values_mean",
                tf.reduce_mean(next_values)
            )
        return rewards + (terminals * self.gamma * next_values)
