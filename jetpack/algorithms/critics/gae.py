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
        values, thermometer = self.gradient_update(
            observations,
            actions,
            rewards,
            lengths
        )
        delta_v = (
            rewards + values[:, 1:] * self.gamma -
            values[:, :(-1)]
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

