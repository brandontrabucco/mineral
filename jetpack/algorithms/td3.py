"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base
from jetpack.core.policy import Policy
from jetpack.core.qf import QF


class TD3(Base):

    def __init__(
        self,
        policy: Policy,
        qf1: QF,
        qf2: QF,
        target_policy: Policy,
        target_qf1: QF,
        target_qf2: QF,
        clip_radius=1.0,
        sigma=1.0,
        gamma=1.0,
        actor_delay=1,
        monitor=None,
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_policy = target_policy
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.clip_radius = clip_radius
        self.sigma = sigma
        self.gamma = gamma
        self.actor_delay = actor_delay
        self.iteration = 0
        self.monitor = monitor

    def get_target_values(
        self,
        rewards,
        next_observations
    ):
        next_actions = self.target_policy.get_deterministic_actions(
            next_observations
        )
        epsilon = tf.clip_by_value(
            self.sigma * tf.random.normal(
                next_actions.shape,
                dtype=next_actions.dtype
            ),
            -self.clip_radius,
            self.clip_radius
        )
        noisy_next_actions = next_actions + epsilon
        next_target_qvalues1 = self.target_qf1.get_qvalues(
            next_observations, 
            noisy_next_actions
        )
        next_target_qvalues2 = self.target_qf2.get_qvalues(
            next_observations, 
            noisy_next_actions
        )
        minimum_qvalues = tf.minimum(
            next_target_qvalues1,
            next_target_qvalues2
        )
        if self.monitor is not None:
            self.monitor.record("next_target_qvalues1_mean", tf.reduce_mean(next_target_qvalues1))
            self.monitor.record("next_target_qvalues2_mean", tf.reduce_mean(next_target_qvalues2))
            self.monitor.record("minimum_qvalues_mean", tf.reduce_mean(minimum_qvalues))
        return rewards + (self.gamma * minimum_qvalues)

    def update_qf1(
        self,
        observations, 
        actions,
        target_values
    ):
        with tf.GradientTape() as tape_qf1:
            qvalues1 = self.qf1.get_qvalues(
                observations, 
                actions
            )
            loss_qf1 = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    qvalues1
                )
            )
            self.qf1.minimize(
                loss_qf1,
                tape_qf1
            )
            if self.monitor is not None:
                self.monitor.record("loss_qf1", loss_qf1)
                self.monitor.record("qvalues1_mean", tf.reduce_mean(qvalues1))

    def update_qf2(
        self,
        observations, 
        actions,
        target_values
    ):
        with tf.GradientTape() as tape_qf2:
            qvalues2 = self.qf2.get_qvalues(
                observations, 
                actions
            )
            loss_qf2 = tf.reduce_mean(
                tf.losses.mean_squared_error(
                    target_values,
                    qvalues2
                )
            )
            self.qf2.minimize(
                loss_qf2,
                tape_qf2
            )
            if self.monitor is not None:
                self.monitor.record("loss_qf2", loss_qf2)
                self.monitor.record("qvalues2_mean", tf.reduce_mean(qvalues2))

    def update_policy(
        self,
        observations
    ):
        with tf.GradientTape() as tape_policy:
            policy_actions = self.policy.get_deterministic_actions(
                observations
            )
            policy_qvalues1 = self.qf1.get_qvalues(
                observations,
                policy_actions
            )
            policy_qvalues2 = self.qf2.get_qvalues(
                observations,
                policy_actions
            )
            loss_policy = -1.0 * (
                tf.reduce_mean(policy_qvalues1)
            )
            self.policy.minimize(
                loss_policy,
                tape_policy
            )
            if self.monitor is not None:
                self.monitor.record("loss_policy", loss_policy)
                self.monitor.record("policy_qvalues1_mean", tf.reduce_mean(policy_qvalues1))
                self.monitor.record("policy_qvalues2_mean", tf.reduce_mean(policy_qvalues2))

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        next_observations
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        target_values = self.get_target_values(
            rewards,
            next_observations
        )
        if self.monitor is not None:
            self.monitor.record("rewards_mean", tf.reduce_mean(rewards))
            self.monitor.record("target_values_mean", tf.reduce_mean(target_values))
        self.update_qf1(
            observations, 
            actions,
            target_values
        )
        self.update_qf2(
            observations, 
            actions,
            target_values
        )
        if self.iteration % self.actor_delay == 0:
            self.update_policy(
                observations
            )
            self.target_policy.soft_update(
                self.policy.get_weights()
            )
            self.target_qf1.soft_update(
                self.qf1.get_weights()
            )
            self.target_qf2.soft_update(
                self.qf2.get_weights()
            )

