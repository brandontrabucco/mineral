"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.importance_sampling import ImportanceSampling


class PPO(ImportanceSampling):

    def __init__(
        self,
        *args,
        epsilon=0.1,
        alpha=0.0,
        **kwargs
    ):
        ImportanceSampling.__init__(self, *args, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha

    def update_actor(
        self,
        observations,
        actions,
        returns,
        terminals
    ):
        if self.iteration - self.last_old_update_iteration >= self.old_update_every:
            self.last_old_update_iteration = self.iteration
            self.old_policy.set_weights(self.policy.get_weights())

        def loss_function():
            ratio = tf.exp(
                self.policy.get_log_probs(
                    actions, observations[:, :(-1), ...]) - self.old_policy.get_log_probs(
                        actions, observations[:, :(-1), ...]))
            sampled_actions = self.policy.sample(
                observations[:, :(-1), ...])
            sampled_log_probs = self.policy.get_log_probs(
                sampled_actions,
                observations[:, :(-1), ...])
            policy_loss = tf.reduce_mean(
                self.alpha * sampled_log_probs - tf.minimum(
                    returns * ratio,
                    returns * tf.clip_by_value(
                        ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)))
            self.record(
                "policy_loss", policy_loss)
            self.record(
                "policy_log_probs_mean",
                tf.reduce_mean(sampled_log_probs))
            self.record(
                "advantages_max",
                tf.reduce_max(returns))
            self.record(
                "advantages_min",
                tf.reduce_min(returns))
            self.record(
                "advantages_mean",
                tf.reduce_mean(returns))
            return policy_loss
        self.policy.minimize(
            loss_function,
            observations[:, :(-1), ...])
