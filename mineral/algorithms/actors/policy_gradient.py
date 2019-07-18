"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor import Actor
from mineral import discounted_sum


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
            log_probs = self.policy.get_log_probs(
                actions,
                observations[:, :(-1), :]
            )
            loss_policy = -1.0 * tf.reduce_mean(
                returns * log_probs
            )
            if self.monitor is not None:
                self.monitor.record(
                    "log_probs_policy_mean",
                    tf.reduce_mean(log_probs)
                )
                self.monitor.record(
                    "log_probs_policy_max",
                    tf.reduce_max(log_probs)
                )
                self.monitor.record(
                    "log_probs_policy_min",
                    tf.reduce_min(log_probs)
                )
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
        returns = discounted_sum(rewards, self.gamma)
        returns = returns - tf.reduce_mean(returns)
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "returns_max",
                tf.reduce_max(returns)
            )
            self.monitor.record(
                "returns_min",
                tf.reduce_min(returns)
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

