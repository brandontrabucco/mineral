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
            loss_policy = tf.reduce_mean(
                self.alpha * policy_log_probs - returns)
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy)
                self.monitor.record(
                    "policy_log_probs_mean",
                    tf.reduce_mean(policy_log_probs))
                self.monitor.record(
                    "returns_max",
                    tf.reduce_max(returns))
                self.monitor.record(
                    "returns_min",
                    tf.reduce_min(returns))
                self.monitor.record(
                    "returns_mean",
                    tf.reduce_mean(returns))
            return loss_policy
        self.policy.minimize(
            loss_function, observations)
