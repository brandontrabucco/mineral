"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.policy_gradient import PolicyGradient
from mineral.algorithms.actors.actor import Actor


class ActorCritic(PolicyGradient):

    def __init__(
        self,
        policy,
        critic,
        gamma=1.0,
        actor_delay=1,
        monitor=None,
    ):
        PolicyGradient.__init__(
            self,
            policy,
            gamma=gamma,
            monitor=monitor,
        )
        self.critic = critic
        self.actor_delay = actor_delay

    def update_algorithm(
        self, 
        observations,
        actions,
        rewards,
        terminals
    ):
        self.critic.gradient_update(
            observations,
            actions,
            rewards,
            terminals
        )
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
        if self.iteration % self.actor_delay == 0:
            self.update_actor(
                observations,
                actions,
                returns,
                terminals
            )


