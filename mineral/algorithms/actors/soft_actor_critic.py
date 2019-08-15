"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.ddpg import DDPG


class SoftActorCritic(DDPG):

    def __init__(
        self,
        *args,
        alpha=1.0,
        **kwargs
    ):
        DDPG.__init__(
            self,
            *args,
            **kwargs)
        self.alpha = alpha

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
            policy_log_probs = self.policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...],
                training=True)
            advantages = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals)
            policy_loss = tf.reduce_mean(
                self.alpha * policy_log_probs - advantages)
            self.record(
                "policy_loss",
                policy_loss)
            self.record(
                "policy_log_probs_mean",
                tf.reduce_mean(policy_log_probs))
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
            loss_function, observations[:, :(-1), ...])
