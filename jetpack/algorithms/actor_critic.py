"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base


class ActorCritic(Base):

    def __init__(
        self,
        policy,
        critic,
        gamma=1.0,
        actor_delay=1,
        monitor=None,
    ):
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.actor_delay = actor_delay
        self.monitor = monitor
        self.iteration = 0

    def update_policy(
        self,
        observations,
        actions,
        returns
    ):
        with tf.GradientTape() as tape_policy:
            loss_policy = tf.reduce_mean(
                returns * self.policy.get_log_probs(
                    observations[:, :(-1), :],
                    actions
                )
            )
            self.policy.minimize(
                loss_policy,
                tape_policy
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    tf.reduce_mean(loss_policy)
                )

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        lengths
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        returns = self.critic.gradient_update_return_weights(
            observations,
            actions,
            rewards,
            lengths
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "returns_mean",
                tf.reduce_mean(returns)
            )
        if self.iteration % self.actor_delay == 0:
            self.update_policy(
                observations,
                actions,
                returns
            )


