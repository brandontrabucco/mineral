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
            **kwargs)
        self.master_old_policy = old_policy
        self.worker_old_policy = old_policy.clone()
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
            self.worker_old_policy.set_weights(self.worker_policy.get_weights())

        def loss_function():
            ratio = tf.exp(
                self.worker_policy.get_log_probs(
                    actions,
                    observations[:, :(-1), ...],
                    training=True) - self.worker_old_policy.get_log_probs(
                        actions,
                        observations[:, :(-1), ...],
                        training=True))
            policy_loss = -1.0 * tf.reduce_mean(
                returns * ratio)
            self.record(
                "policy_loss",
                policy_loss)
            return policy_loss
        self.worker_policy.minimize(
            loss_function,
            observations[:, :(-1), ...])

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        self.master_old_policy.copy_to(self.worker_old_policy)
        ActorCritic.update_algorithm(
            self,
            observations,
            actions,
            rewards,
            terminals)
        self.worker_old_policy.copy_to(self.master_old_policy)
