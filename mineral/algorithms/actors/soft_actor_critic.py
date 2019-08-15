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
            returns = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals)
            policy_loss = tf.reduce_mean(
                self.alpha * policy_log_probs - returns)
            self.record(
                "policy_loss",
                policy_loss)
            self.record(
                "policy_log_probs_mean",
                tf.reduce_mean(policy_log_probs))
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
            loss_function, observations)
