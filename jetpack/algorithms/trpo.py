"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
import jetpack as jp
from jetpack.algorithms.actor_critic import ActorCritic


class TRPO(ActorCritic):

    def __init__(
        self,
        policy,
        old_policy,
        critic,
        gamma=1.0,
        delta=1.0,
        actor_delay=1,
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
        self.delta = delta

    def update_policy(
        self,
        observations,
        actions,
        returns
    ):

        with tf.GradientTape() as tape_policy:

            means = self.policy.get_deterministic_actions(
                observations
            )

            grads, vars = zip(*tf.gradients(
                means,
                self.policy.trainable_variables
            ))

            y = grads

            Jy = jp.jacobian_vector_product(
                means,
                self.policy.trainable_variables,
                y
            )

            J_TJy = tf.gradients(
                means,
                self.policy.trainable_variables,
                grad_ys=Jy
            )

            ratio = tf.exp(
                self.policy.get_log_probs(
                    observations[:, :(-1), :],
                    actions
                ) - self.old_policy.get_log_probs(
                    observations[:, :(-1), :],
                    actions
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
            self.policy.minimize(
                loss_policy,
                tape_policy
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    tf.reduce_mean(loss_policy)
                )
        if self.iteration % self.old_policy_delay == 0:
            self.old_policy.soft_update(
                self.policy.get_weights()
            )
