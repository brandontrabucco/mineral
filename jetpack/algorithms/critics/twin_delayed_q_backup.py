"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class TwinDelayedQBackup(Critic):

    def __init__(
        self,
        q_backup1,
        q_backup2,
        monitor=None,
    ):
        self.q_backup1 = q_backup1
        self.q_backup2 = q_backup2
        self.qf = q_backup1.qf
        self.iteration = 0
        self.monitor = monitor

    def get_target_values(
        self,
        rewards,
        next_observations
    ):
        target_values1 = self.q_backup1.get_target_values(
            rewards,
            next_observations
        )
        target_values2 = self.q_backup2.get_target_values(
            rewards,
            next_observations
        )
        return tf.minimum(target_values1, target_values2)

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
        self.q_backup1.target_qf.soft_update(
            self.q_backup1.qf.get_weights()
        )
        self.q_backup2.target_qf.soft_update(
            self.q_backup2.qf.get_weights()
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

