"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.actors.actor import Actor


class DDPG(Actor):

    def __init__(
        self,
        policy,
        critic,
        target_policy,
        actor_delay=1,
        monitor=None,
    ):
        self.policy = policy
        self.critic = critic
        self.target_policy = target_policy
        self.actor_delay = actor_delay
        self.monitor = monitor
        self.iteration = 0

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            policy_actions = self.policy.get_stochastic_actions(
                observations[:, :(-1), :]
            )
            policy_advantages = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals
            )
            loss_policy = -1.0 * (
                tf.reduce_mean(policy_advantages)
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
                self.monitor.record(
                    "policy_advantages_mean",
                    tf.reduce_mean(policy_advantages)
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
        self.critic.gradient_update(
            observations,
            actions,
            rewards,
            terminals
        )
        if self.iteration % self.actor_delay == 0:
            self.update_actor(
                observations,
                actions,
                rewards,
                terminals
            )
            self.target_policy.soft_update(
                self.policy.get_weights()
            )
