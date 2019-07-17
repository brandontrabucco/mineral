"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.value_learning import ValueLearning


class GAE(ValueLearning):

    def __init__(
        self,
        vf,
        target_vf,
        gamma=1.0,
        lamb=1.0,
        monitor=None,
    ):
        ValueLearning.__init__(
            self,
            vf,
            target_vf,
            gamma=gamma,
            monitor=monitor
        )
        self.lamb = lamb

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        values = self.vf.get_values(
            observations
        )[:, :, 0]
        delta_v = (
            terminals[:, :(-1)] * rewards -
            terminals[:, :(-1)] * values[:, :(-1)] +
            terminals[:, 1:] * values[:, 1:] * self.gamma
        )
        weights = tf.tile(
            [[self.gamma * self.lamb]],
            [1, tf.shape(delta_v)[1]]
        )
        weights = tf.math.cumprod(weights, axis=1, exclusive=True)
        advantages = tf.math.cumsum(delta_v * weights, axis=1, reverse=True) / weights
        if self.monitor is not None:
            self.monitor.record(
                "advantages_mean",
                tf.reduce_mean(advantages)
            )
        return advantages

