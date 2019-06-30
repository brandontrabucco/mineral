"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.actor_critic import ActorCritic
from jetpack.line_search import line_search


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
            def loss_function(
                policy
            ):
                kl = policy.get_kl_divergence(
                    policy,
                    self.old_policy
                )
                expected_return = tf.reduce_mean(
                    returns * policy.get_log_probs(
                        observations[:, :(-1), :],
                        actions
                    )
                )
                return -1.0 * expected_return + (
                    0.0 if kl < self.delta else float("inf")
                )
            loss_policy = loss_function(
                self.policy
            )
            grad = tape_policy.gradient(
                loss_policy,
                self.policy.trainable_variables
            )
            grad, sAs = self.policy.naturalize(
                observations,
                grad
            )
            grad = line_search(
                loss_function,
                self.policy,
                grad,
                tf.math.sqrt(self.delta / sAs)
            )
            self.policy.apply_gradients(
                grad
            )
            self.old_policy.set_weights(
                self.policy.get_weights()
            )
            if self.monitor is not None:
                self.monitor.record(
                    "loss_policy",
                    loss_policy
                )
