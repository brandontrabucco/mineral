"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor_critic import ActorCritic


class DDPG(ActorCritic):

    def __init__(
        self,
        policy,
        target_policy,
        critic,
        **kwargs
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            **kwargs)
        self.target_policy = target_policy

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
            returns = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals)
            policy_loss = -1.0 * (
                tf.reduce_mean(returns))
            self.record(
                "policy_loss",
                policy_loss)
            self.record(
                "returns_max",
                tf.reduce_max(returns))
            self.record(
                "returns_min",
                tf.reduce_min(returns))
            self.record(
                "returns_mean",
                tf.reduce_mean(returns))
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
        self.target_policy.soft_update(
            self.policy.get_weights())
