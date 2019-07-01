"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.value_learning import ValueLearning


class TwinDelayedValueLearning(ValueLearning):

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
        observations
    ):
        values1 = self.value_backup1.get_values(
            observations
        )
        values2 = self.value_backup2.get_values(
            observations
        )
        return 0.5 * (values1 + values2)

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
        target_values,
        terminals
    ):
        values1 = self.value_backup1.update_vf(
            observations,
            target_values,
            terminals
        )
        values2 = self.value_backup2.update_vf(
            observations,
            target_values,
            terminals
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
