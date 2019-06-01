"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.base import Base
from jetpack.networks.policy import Policy
from jetpack.networks.qf import QF


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

    def get_target_values(
        self,
        rewards,
        next_observations,
    ):
        next_actions = self.target_policy.get_deterministic_actions(
            next_observations,
        )
        epsilon = tf.clip_by_value(
            self.sigma * tf.random.normal(next_actions.shape),
            -self.clip_radius,
            self.clip_radius,
        )
        noisy_next_actions = next_actions + epsilon
        next_target_qvalue1 = self.target_qf1.get_qvalues(
            next_observations, 
            noisy_next_actions,
        )
        next_target_qvalue2 = self.target_qf2.get_qvalues(
            next_observations, 
            noisy_next_actions,
        )
        minimum_qvalue = tf.minimum(
            next_target_qvalue1, 
            next_target_qvalue2,
        )
        return rewards + (self.gamma * minimum_qvalue)

    def update_qf1(
        self,
        observations, 
        actions,
        target_values, 
    ):
        with tf.GradientTape() as tape_qf1:
            qvalue1 = self.qf1.get_qvalues(
                observations, 
                actions,
            )
            loss_qf1 = tf.losses.mean_squared_error(
                target_values, 
                qvalue1,
            )
        gradients_qf1 = tape_qf1.gradient(
            loss_qf1, 
            self.qf1.trainable_variables,
        )
        self.qf1.apply_gradients(gradients_qf1)

    def update_qf2(
        self,
        observations, 
        actions,
        target_values, 
    ):
        with tf.GradientTape() as tape_qf2:
            qvalue2 = self.qf2.get_qvalues(
                observations, 
                actions,
            )
            loss_qf2 = tf.losses.mean_squared_error(
                target_values, 
                qvalue2,
            )
        gradients_qf2 = tape_qf2.gradient(
            loss_qf2, 
            self.qf2.trainable_variables,
        )
        self.qf2.apply_gradients(gradients_qf2)

    def update_policy(
        self,
        observations,
    ):
        with tf.GradientTape() as tape_policy:
            policy_actions = self.policy.get_deterministic_actions(
                observations,
            )
            policy_qvalue1 = self.qf1.get_qvalues(
                observations,
                policy_actions,
            )
            policy_qvalue2 = self.qf2.get_qvalues(
                observations,
                policy_actions,
            )
            loss_policy = -0.5 * (
                tf.reduce_mean(policy_qvalue1) + 
                tf.reduce_mean(policy_qvalue2)
            )
        gradients_policy = tape_policy.gradient(
            loss_policy, 
            self.policy.trainable_variables,
        )
        self.policy.apply_gradients(gradients_policy)

    def gradient_update(
        self, 
        observations,
        actions,
        rewards,
        next_observations,
    ):
        self.iteration += 1
        target_values = self.get_target_values(
            rewards,
            next_observations,
        )
        self.update_qf1(
            observations, 
            actions,
            target_values, 
        )
        self.update_qf2(
            observations, 
            actions,
            target_values, 
        )
        if self.iteration % self.actor_delay == 0:
            self.update_policy(observations)
            self.target_policy.soft_update(self.policy.get_weights())
            self.target_qf1.soft_update(self.qf1.get_weights())
            self.target_qf2.soft_update(self.qf2.get_weights())
