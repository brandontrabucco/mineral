"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base


class PolicyGradient(Base):

    def __init__(
        self,
        policy,
        gamma=1.0,
        monitor=None,
    ):
        self.policy = policy
        self.gamma = gamma
        self.monitor = monitor
        self.iteration = 0

    def update_policy(
        self,
        observations,
        actions,
        returns
    ):
        def loss_function():
            loss_policy = -1.0 * tf.reduce_mean(
                returns * self.policy.get_log_probs(
                    observations[:, :(-1), :],
                    actions
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
            return loss_policy
        self.policy.minimize(
            loss_function,
            observations
        )

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
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "returns_mean",
                tf.reduce_mean(returns)
            )
        self.update_policy(
            observations,
            actions,
            returns
        )

