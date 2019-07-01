"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.value_regression import ValueRegression


class GAE(ValueRegression):

    def __init__(
        self,
        vf,
        gamma=1.0,
        lamb=1.0,
        monitor=None,
    ):
        ValueRegression.__init__(
            self,
            vf,
            gamma=gamma,
            monitor=monitor
        )
        self.lamb = lamb

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        lengths
    ):
        thermometer = self.gradient_update(
            observations,
            actions,
            rewards,
            lengths
        )
        values = self.vf.get_values(
            observations
        )[:, :, 0]
        delta_v = (
            thermometer[:, :(-1)] * rewards -
            thermometer[:, :(-1)] * values[:, :(-1)] +
            thermometer[:, 1:] * values[:, 1:] * self.gamma
        )
        weights = tf.tile(
            [[self.gamma * self.lamb]],
            [1, tf.shape(delta_v)[1]]
        )
        weights = tf.math.cumprod(
            weights,
            axis=1,
            exclusive=True
        )
        advantages = tf.math.cumsum(
            delta_v * weights
        ) / weights
        if self.monitor is not None:
            self.monitor.record(
                "advantages_mean",
                tf.reduce_mean(advantages)
            )
        return advantages

