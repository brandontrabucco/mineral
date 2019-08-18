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
        old_update_after=1,
        **kwargs
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            **kwargs)
        self.old_policy = old_policy
        self.old_update_every = old_update_every
        self.old_update_after = old_update_after
        self.last_old_update_iteration = 0

    def update_actor(
        self,
        observations,
        actions,
        returns,
        terminals
    ):
        if (self.iteration > self.old_update_after and
                self.iteration - self.last_old_update_iteration >= self.old_update_every):
            self.last_old_update_iteration = self.iteration
            self.old_policy.set_weights(self.policy.get_weights())

        def loss_function():
            ratio = tf.exp(
                self.policy.get_log_probs(
                    actions, observations[:, :(-1), ...]) - self.old_policy.get_log_probs(
                        actions, observations[:, :(-1), ...]))
            policy_loss = -1.0 * tf.reduce_mean(
                returns * ratio)
            self.record("policy_loss", policy_loss)
            return policy_loss
        self.policy.minimize(
            loss_function,
            observations[:, :(-1), ...])
