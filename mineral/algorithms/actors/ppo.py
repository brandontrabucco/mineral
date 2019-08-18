"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.importance_sampling import ImportanceSampling


class PPO(ImportanceSampling):

    def __init__(
        self,
        *args,
        epsilon=0.2,
        **kwargs
    ):
        ImportanceSampling.__init__(self, *args, **kwargs)
        self.epsilon = epsilon

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        if (self.iteration >= self.old_update_after and
                self.iteration - self.last_old_update_iteration >= self.old_update_every):
            self.last_old_update_iteration = self.iteration
            self.old_policy.set_weights(self.policy.get_weights())

        def loss_function():
            advantages = self.critic.get_advantages(
                observations,
                actions,
                rewards,
                terminals)
            ratio = tf.exp(
                self.policy.get_log_probs(
                    actions, observations[:, :(-1), ...]) - self.old_policy.get_log_probs(
                    actions, observations[:, :(-1), ...]))
            policy_loss = -1.0 * tf.reduce_mean(
                tf.minimum(
                    advantages * ratio,
                    advantages * tf.clip_by_value(
                        ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)))
            self.record(
                "rewards_mean", tf.reduce_mean(rewards))
            self.record(
                "policy_ratio_mean", tf.reduce_mean(ratio))
            self.record(
                "policy_ratio_max", tf.reduce_max(ratio))
            self.record(
                "policy_ratio_min", tf.reduce_min(ratio))
            self.record(
                "policy_loss", policy_loss)
            return policy_loss
        self.policy.minimize(
            loss_function,
            observations[:, :(-1), ...])
