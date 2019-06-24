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
        target_vf.set_weights(vf.get_weights())

    def get_target_values(
        self,
        rewards,
        next_observations
    ):
        next_target_values = self.target_vf.get_qvalues(
            next_observations
        )
        if self.monitor is not None:
            self.monitor.record(
                "next_target_values_mean",
                tf.reduce_mean(next_target_values)
            )
        return rewards + (self.gamma * next_target_values)

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
        with tf.GradientTape() as tape_vf:
            values = self.vf.get_values(
                observations
            )
            loss_vf = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    values
                )
            )
            self.vf.minimize(
                loss_vf,
                tape_vf
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
            self.target_vf.soft_update(
                self.vf.get_weights()
            )

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        next_observations
    ):
        self.gradient_update(
            observations,
            actions,
            rewards,
            next_observations
        )
        next_values = self.vf.get_qvalues(
            next_observations
        )
        if self.monitor is not None:
            self.monitor.record(
                "next_values_mean",
                tf.reduce_mean(next_values)
            )
        return rewards + (self.gamma * next_values)
