"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class ValueRegression(Critic):

    def __init__(
        self,
        vf,
        gamma=1.0,
        monitor=None,
    ):
        self.vf = vf
        self.gamma = gamma
        self.iteration = 0
        self.monitor = monitor

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        lengths
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        weights = tf.tile(
            [[self.gamma]],
            [1, tf.shape(rewards)[1]]
        )
        weights = tf.math.cumprod(
            weights,
            axis=1,
            exclusive=True
        )
        thermometer = tf.cast(
            tf.range(
                tf.shape(observations)[1]
            )[tf.newaxis, :] < lengths[:, tf.newaxis],
            tf.float32
        )
        returns = tf.math.cumsum(
            rewards * thermometer[:, :(-1)] * weights
        ) / weights
        with tf.GradientTape() as vf_tape:
            values = self.vf.get_values(
                observations,
            )[:, :, 0] * thermometer
            vf_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    returns,
                    values[:, :(-1)]
                )
            )
            self.vf.minimize(
                vf_loss,
                vf_tape
            )
            if self.monitor is not None:
                self.monitor.record(
                    "returns_mean",
                    tf.reduce_mean(returns)
                )
                self.monitor.record(
                    "values_mean",
                    tf.reduce_mean(values)
                )
                self.monitor.record(
                    "vf_loss",
                    vf_loss
                )
            return values, thermometer

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
        return (
            rewards * thermometer[:, :(-1)] +
            values[:, 1:]
        )

