"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.importance_sampling import ImportanceSampling


class PPO(ImportanceSampling):

    def __init__(
        self,
        *args,
        epsilon=1.0,
        **kwargs
    ):
        ImportanceSampling.__init__(
            self,
            *args,
            **kwargs
        )
        self.epsilon = epsilon

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
                    actions,
                    observations[:, :(-1), ...]
                ) - self.old_policy.get_log_probs(
                    actions,
                    observations[:, :(-1), ...]
                )
            )
            loss_policy = -1.0 * tf.reduce_mean(
                tf.minimum(
                    returns * ratio,
                    returns * tf.clip_by_value(
                        ratio, 1 - self.epsilon, 1 + self.epsilon
                    )
                )
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
            return loss_policy
        self.policy.minimize(
            loss_function,
            observations[:, :(-1), ...])
