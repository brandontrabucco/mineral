"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.q_learning import QLearning


class TwinDelayedQLearning(QLearning):

    def __init__(
        self,
        q_backup1,
        q_backup2,
        monitor=None,
    ):
        self.q_backup1 = q_backup1
        self.q_backup2 = q_backup2
        self.iteration = 0
        self.monitor = monitor

    def get_qvalues(
        self,
        observations,
        actions
    ):
        qvalues1 = self.q_backup1.get_qvalues(
            observations,
            actions,
        )
        qvalues2 = self.q_backup2.get_qvalues(
            observations,
            actions,
        )
        return tf.reduce_mean(qvalues1, qvalues2)

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        target_values1 = self.q_backup1.get_target_values(
            rewards,
            next_observations,
            terminals
        )
        target_values2 = self.q_backup2.get_target_values(
            rewards,
            next_observations,
            terminals
        )
        return tf.minimum(target_values1, target_values2)

    def update_qf(
        self,
        observations,
        actions,
        target_values
    ):
        qvalues1 = self.q_backup1.update_qf(
            observations,
            actions,
            target_values
        )
        qvalues2 = self.q_backup2.update_qf(
            observations,
            actions,
            target_values
        )
        return qvalues1, qvalues2

    def soft_update(
        self
    ):
        self.q_backup1.target_qf.soft_update(
            self.q_backup1.qf.get_weights()
        )
        self.q_backup2.target_qf.soft_update(
            self.q_backup2.qf.get_weights()
        )

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        qvalues1, qvalues2 = self.gradient_update(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )
        return tf.reduce_mean(qvalues1, qvalues2)

