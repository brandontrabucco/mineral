"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor import Actor
from mineral import discounted_sum


class PolicyGradient(Actor):

    def __init__(
        self,
        policy,
        gamma=1.0,
        **kwargs
    ):
        Actor.__init__(self, **kwargs)
        self.policy = policy
        self.gamma = gamma

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            returns = discounted_sum(rewards, self.gamma)
            advantages = returns - tf.reduce_mean(returns)
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

