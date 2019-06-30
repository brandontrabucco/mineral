"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base


class DDPG(Base):

    def __init__(
        self,
        policy,
        q_backup,
        target_policy,
        actor_delay=1,
        monitor=None,
    ):
        self.policy = policy
        self.q_backup = q_backup
        self.target_policy = target_policy
        self.actor_delay = actor_delay
        self.monitor = monitor
        self.iteration = 0

    def update_policy(
        self,
        observations
    ):
        with tf.GradientTape() as tape_policy:
            policy_actions = self.policy.get_deterministic_actions(
                observations
            )
            policy_qvalues = self.q_backup.get_qvalues(
                observations,
                policy_actions
            )
            loss_policy = -1.0 * (
                tf.reduce_mean(policy_qvalues)
            )
            self.policy.minimize(
                loss_policy,
                tape_policy
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
                self.monitor.record(
                    "policy_qvalues_mean",
                    tf.reduce_mean(policy_qvalues)
                )

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        self.q_backup.gradient_update(
            observations,
            actions,
            rewards,
            next_observations,
            terminals
        )
        if self.iteration % self.actor_delay == 0:
            self.update_policy(
                observations
            )
            self.target_policy.soft_update(
                self.policy.get_weights()
            )
