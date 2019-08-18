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
            **kwargs)
        self.critic = critic

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            advantages = self.critic.get_advantages(
                observations,
                actions,
                rewards,
                terminals)
            log_probs = self.policy.get_log_probs(
                actions,
                observations[:, :(-1), ...])
            policy_loss = -1.0 * tf.reduce_mean(
                advantages * log_probs)
            self.record(
                "rewards_mean", tf.reduce_mean(rewards))
            self.record(
                "log_probs_policy_mean", tf.reduce_mean(log_probs))
            self.record(
                "log_probs_policy_max", tf.reduce_max(log_probs))
            self.record(
                "log_probs_policy_min", tf.reduce_min(log_probs))
            self.record(
                "policy_loss", policy_loss)
            return policy_loss
        self.policy.minimize(
            loss_function, observations[:, :(-1), ...])
