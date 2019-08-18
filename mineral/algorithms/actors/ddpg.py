"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor_critic import ActorCritic


class DDPG(ActorCritic):

    def __init__(
        self,
        policy,
        critic,
        **kwargs
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            **kwargs)

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            policy_actions = self.policy.sample(
                observations[:, :(-1), ...],
                training=True)
            advantages = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals)
            policy_loss = (-1.0) * tf.reduce_mean(advantages)
            self.record(
                "policy_loss",
                policy_loss)
            self.record(
                "advantages_max",
                tf.reduce_max(advantages))
            self.record(
                "advantages_min",
                tf.reduce_min(advantages))
            self.record(
                "advantages_mean",
                tf.reduce_mean(advantages))
            self.record(
                "rewards_max",
                tf.reduce_max(rewards))
            self.record(
                "rewards_min",
                tf.reduce_min(rewards))
            self.record(
                "rewards_mean",
                tf.reduce_mean(rewards))
            return policy_loss
        self.policy.minimize(
            loss_function,
            observations[:, :(-1), ...])

    def update_algorithm(
        self, 
        observations,
        actions,
        rewards,
        terminals
    ):
        self.update_actor(
            observations,
            actions,
            rewards,
            terminals)
