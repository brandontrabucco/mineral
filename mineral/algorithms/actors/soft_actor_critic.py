"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.ddpg import DDPG


class SoftActorCritic(DDPG):

    def __init__(
        self,
        *args,
        entropy=-1.0,
        entropy_optimizer_class=tf.keras.optimizers.Adam,
        entropy_optimizer_kwargs={},
        **kwargs
    ):
        DDPG.__init__(
            self,
            *args,
            **kwargs)
        self.alpha = tf.Variable(1.0)
        self.entropy = entropy
        self.entropy_optimizer = entropy_optimizer_class(
            **entropy_optimizer_kwargs)

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            policy_actions = self.policy.sample(
                observations[:, :(-1), ...])
            policy_log_probs = self.policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...])
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

        def entropy_loss_function():
            policy_actions = self.policy.sample(
                observations[:, :(-1), ...])
            policy_log_probs = self.policy.get_log_probs(
                policy_actions,
                observations[:, :(-1), ...])
            loss_entropy = -self.alpha * (
                policy_log_probs + self.entropy)
            return tf.reduce_mean(loss_entropy)
        self.entropy_optimizer.minimize(
            entropy_loss_function, self.alpha)
