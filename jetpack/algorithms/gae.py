"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critic import Critic


class GAE(Critic):

    def __init__(
        self,
        vf,
        gamma=1.0,
        lamb=1.0,
        monitor=None,
    ):
        self.vf = vf
        self.gamma = gamma
        self.lamb = lamb
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
            observations.dtype
        )
        returns = tf.math.cumsum(
            rewards * thermometer[:, :(-1)] * weights
        ) / weights
        with tf.GradientTape() as vf_tape:
            values = self.vf.get_values(
                observations
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
            return values

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        lengths
    ):
        values = self.gradient_update(
            observations,
            actions,
            rewards,
            lengths
        )
        delta_v = (
            rewards + values[:, 0:, 0] * self.gamma -
            values[:, :(-1), 0]
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

