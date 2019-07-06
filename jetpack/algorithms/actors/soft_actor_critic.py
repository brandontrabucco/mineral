"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.actors.ddpg import DDPG


class SoftActorCritic(DDPG):

    def __init__(
        self,
        policy,
        critic,
        target_policy,
        actor_delay=1,
        monitor=None,
    ):
        DDPG.__init__(
            self,
            policy,
            critic,
            target_policy,
            actor_delay=actor_delay,
            monitor=monitor,
        )

    def update_actor(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        def loss_function():
            policy_actions = self.policy.get_stochastic_actions(
                observations[:, :(-1), :]
            )
            policy_log_probs = self.policy.get_log_probs(
                observations[:, :(-1), :],
                policy_actions
            )
            policy_advantages = self.critic.get_advantages(
                observations,
                policy_actions,
                rewards,
                terminals
            )
            loss_policy = tf.reduce_mean(
                policy_log_probs - policy_advantages
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
                self.monitor.record(
                    "policy_advantages_mean",
                    tf.reduce_mean(policy_advantages)
                )
                self.monitor.record(
                    "policy_log_probs_mean",
                    tf.reduce_mean(policy_log_probs)
                )
            return loss_policy
        self.policy.minimize(
            loss_function,
            observations
        )