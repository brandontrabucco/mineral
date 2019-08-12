"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.policy_gradient import PolicyGradient
from mineral.algorithms.actors.actor import Actor


class ActorCritic(PolicyGradient):

    def __init__(
        self,
        policy,
        critic,
        **kwargs
    ):
        PolicyGradient.__init__(
            self,
            policy,
            **kwargs
        )
        self.critic = critic

    def update_algorithm(
        self, 
        observations,
        actions,
        rewards,
        terminals
    ):
        returns = self.critic.get_advantages(
            observations,
            actions,
            rewards,
            terminals
        )
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


