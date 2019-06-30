"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.q_learning import QLearning


class SoftQLearning(QLearning):

    def __init__(
        self,
        qf,
        target_policy,
        target_qf,
        gamma=1.0,
        clip_radius=1.0,
        sigma=1.0,
        monitor=None,
    ):
        QLearning.__init__(
            self,
            qf,
            target_policy,
            target_qf,
            gamma=gamma,
            clip_radius=clip_radius,
            sigma=sigma,
            monitor=monitor,
        )

    def get_target_values(
        self,
        rewards,
        next_observations,
        terminals
    ):
        next_actions = self.target_policy.get_deterministic_actions(
            next_observations
        )
        next_target_log_probs = self.target_policy.get_log_probs(
            next_observations,
            next_actions
        )
        epsilon = tf.clip_by_value(
            self.sigma * tf.random.normal(
                next_actions.shape,
                dtype=tf.float32
            ),
            -self.clip_radius,
            self.clip_radius
        )
        noisy_next_actions = next_actions + epsilon
        next_target_qvalues = self.target_qf.get_qvalues(
            next_observations,
            noisy_next_actions
        )
        target_values = rewards + (
            terminals * self.gamma * (
                next_target_qvalues - next_target_log_probs
            )
        )
        if self.monitor is not None:
            self.monitor.record(
                "rewards_mean",
                tf.reduce_mean(rewards)
            )
            self.monitor.record(
                "next_target_qvalues_mean",
                tf.reduce_mean(next_target_qvalues)
            )
            self.monitor.record(
                "next_target_log_probs_mean",
                tf.reduce_mean(next_target_log_probs)
            )
            self.monitor.record(
                "targets_mean",
                tf.reduce_mean(target_values)
            )
        return target_values