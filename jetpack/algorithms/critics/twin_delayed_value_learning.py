"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.q_learning import QLearning


class TwinDelayedQLearning(QLearning):

    def __init__(
        self,
        value_backup1,
        value_backup2,
        monitor=None,
    ):
        self.value_backup1 = value_backup1
        self.value_backup2 = value_backup2
        self.iteration = 0
        self.monitor = monitor

    def get_values(
        self,
        observations,
        actions
    ):
        values1 = self.value_backup1.get_values(
            observations,
            actions,
        )
        values2 = self.value_backup2.get_values(
            observations,
            actions,
        )
        return tf.reduce_mean(values1, values2)

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        target_values1 = self.value_backup1.get_target_values(
            rewards,
            next_observations,
            terminals
        )
        target_values2 = self.value_backup2.get_target_values(
            rewards,
            next_observations,
            terminals
        )
        return tf.minimum(target_values1, target_values2)

    def update_vf(
        self,
        observations,
        actions,
        target_values
    ):
        values1 = self.value_backup1.update_vf(
            observations,
            actions,
            target_values
        )
        values2 = self.value_backup2.update_vf(
            observations,
            actions,
            target_values
        )
        return values1, values2

    def soft_update(
        self
    ):
        self.value_backup1.target_vf.soft_update(
            self.value_backup1.vf.get_weights()
        )
        self.value_backup2.target_vf.soft_update(
            self.value_backup2.vf.get_weights()
        )

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        values1, values2 = self.gradient_update(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )
        return tf.reduce_mean(values1, values2)
