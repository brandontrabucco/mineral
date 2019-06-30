"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.ddpg import DDPG


class SAC(DDPG):

    def __init__(
        self,
        policy,
        q_backup,
        target_policy,
        actor_delay=1,
        monitor=None,
    ):
        DDPG.__init__(
            policy,
            q_backup,
            target_policy,
            actor_delay=actor_delay,
            monitor=monitor,
        )

    def update_policy(
        self,
        observations
    ):
        with tf.GradientTape() as tape_policy:
            policy_actions = self.policy.get_stochastic_actions(
                observations
            )
            policy_log_probs = self.policy.get_log_probs(
                observations,
                policy_actions
            )
            policy_qvalues = self.q_backup.get_qvalues(
                observations,
                policy_actions
            )
            loss_policy = tf.reduce_mean(
                policy_log_probs - policy_qvalues
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
                self.monitor.record(
                    "policy_log_probs_mean",
                    tf.reduce_mean(policy_log_probs)
                )