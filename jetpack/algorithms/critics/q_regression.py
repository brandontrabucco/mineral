"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class QRegression(Critic):

    def __init__(
        self,
        qf,
        gamma=1.0,
        monitor=None,
    ):
        self.qf = qf
        self.gamma = gamma
        self.iteration = 0
        self.monitor = monitor

    def get_qvalues(
        self,
        observations,
        actions
    ):
        return self.qf.get_qvalues(
            observations,
            actions
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
                tf.shape(observations)[1] - 1
            )[tf.newaxis, :] < lengths[:, tf.newaxis],
            tf.float32
        )
        returns = tf.math.cumsum(
            rewards * thermometer * weights
        ) / weights
        def loss_function(
            *inputs
        ):
            qvalues = self.qf.get_qvalues(
                observations[:, :(-1), :],
                actions
            )[:, :, 0] * thermometer
            qf_loss = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    returns,
                    qvalues
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "qvalues_mean",
                    tf.reduce_mean(qvalues)
                )
                self.monitor.record(
                    "qf_loss",
                    qf_loss
                )
            return qf_loss
        self.qf.minimize(
            loss_function,
            observations[:, :(-1), :],
            actions
        )
        if self.monitor is not None:
            self.monitor.record(
                "returns_mean",
                tf.reduce_mean(returns)
            )

    def gradient_update_return_weights(
        self,
        observations,
        actions,
        rewards,
        lengths
    ):
        self.gradient_update(
            observations,
            actions,
            rewards,
            lengths
        )
        return self.get_qvalues(
            observations,
            actions
        )[:, :, 0]

