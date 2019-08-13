"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.actors.actor_critic import ActorCritic


class ImportanceSampling(ActorCritic):

    def __init__(
        self,
        policy,
        old_policy,
        critic,
        old_update_every=1,
        **kwargs
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            **kwargs
        )
        self.old_policy = old_policy
        self.old_update_every = old_update_every
        self.last_old_update_iteration = 0

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
                returns * ratio
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
