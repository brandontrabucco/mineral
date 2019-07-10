"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.actors.actor_critic import ActorCritic


class PPO(ActorCritic):

    def __init__(
        self,
        policy,
        old_policy,
        critic,
        gamma=1.0,
        epsilon=1.0,
        actor_delay=1,
        old_policy_delay=1,
        monitor=None,
    ):
        ActorCritic.__init__(
            self,
            policy,
            critic,
            gamma=gamma,
            actor_delay=actor_delay,
            monitor=monitor,
        )
        self.old_policy = old_policy
        self.epsilon = epsilon
        self.old_policy_delay = old_policy_delay

    def update_actor(
        self,
        observations,
        actions,
        returns,
        terminals
    ):
        def loss_function():
            ratio = tf.exp(
                self.policy.get_log_probs(
                    actions,
                    observations[:, :(-1), :]
                ) - self.old_policy.get_log_probs(
                    actions,
                    observations[:, :(-1), :]
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
            observations[:, :(-1), :]
        )
        if self.iteration % self.old_policy_delay == 0:
            self.old_policy.set_weights(
                self.policy.get_weights()
            )
