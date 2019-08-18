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
        returns,
        terminals
    ):
        def loss_function():
            log_probs = self.policy.get_log_probs(
                actions,
                observations[:, :(-1), ...],
                training=True)
            policy_loss = -1.0 * tf.reduce_mean(
                returns * log_probs)
            self.record(
                "log_probs_policy_mean",
                tf.reduce_mean(log_probs))
            self.record(
                "log_probs_policy_max",
                tf.reduce_max(log_probs))
            self.record(
                "log_probs_policy_min",
                tf.reduce_min(log_probs))
            self.record(
                "policy_loss",
                policy_loss)
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
        returns = discounted_sum(rewards, self.gamma)
        advantages = returns - tf.reduce_mean(returns)
        self.record(
            "rewards_mean",
            tf.reduce_mean(rewards))
        self.record(
            "returns_max",
            tf.reduce_max(returns))
        self.record(
            "returns_min",
            tf.reduce_min(returns))
        self.record(
            "returns_mean",
            tf.reduce_mean(returns))
        self.update_actor(
            observations,
            actions,
            advantages,
            terminals)

