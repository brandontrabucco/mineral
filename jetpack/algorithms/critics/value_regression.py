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

    def get_values(
        self,
        observations
    ):
        return self.vf.get_values(
            observations
        )[:, :, 0]

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
        if self.monitor is not None:
            self.monitor.record(
                "returns_mean",
                tf.reduce_mean(returns)
            )
        def loss_function():
            values = thermometer * self.vf.get_values(
                observations
            )[:, :, 0]
            vf_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    returns,
                    values[:, :(-1)]
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "values_mean",
                    tf.reduce_mean(values)
                )
                self.monitor.record(
                    "vf_loss",
                    vf_loss
                )
            return vf_loss
        self.vf.minimize(
            loss_function,
            observations
        )
        return thermometer

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
        return (
            thermometer[:, 1:] * values[:, 1:] +
            thermometer[:, :(-1)] * rewards
        )

