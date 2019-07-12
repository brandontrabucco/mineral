"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.actors.actor import Actor


class PolicyGradient(Actor):

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

    def update_actor(
        self,
        observations,
        actions,
        returns,
        terminals
    ):
        def loss_function():
            loss_policy = -1.0 * tf.reduce_mean(
                returns * self.policy.get_log_probs(
                    actions,
                    observations[:, :(-1), :]
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
        terminals
    ):
        Actor.gradient_update(
            self,
            observations,
            actions,
            rewards,
            terminals
        )
        weights = tf.tile(
            [[self.gamma]],
            [1, tf.shape(rewards)[1]]
        )
        weights = tf.math.cumprod(
            weights,
            axis=1,
            exclusive=True
        )
        returns = tf.math.cumsum(
            rewards * weights
        ) / weights
        returns = returns - tf.reduce_mean(returns)
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "returns_mean",
                tf.reduce_mean(returns)
            )
        self.update_actor(
            observations,
            actions,
            returns,
            terminals
        )

