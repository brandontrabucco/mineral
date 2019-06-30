"""Author: Brandon Trabucco, Copyright 2019"""

import gym
import tensorflow as tf
from jetpack.networks.policies.tanh_gaussian_policy import TanhGaussianPolicy
from jetpack.wrappers.normalized_env import NormalizedEnv
from jetpack.line_search import line_search

if __name__ == "__main__":

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = TanhGaussianPolicy(
        [32, 32, 6],
    )

    observations = tf.random.normal([32, 6])
    rewards = tf.ones([32])

    actions = policy.get_stochastic_actions(
        observations
    )

    def loss_function(
        policy
    ):
        return -1.0 * tf.reduce_mean(
            rewards * policy.get_log_probs(
                observations,
                actions
            )
        )

    delta = 0.1

    for i in range(100):

        with tf.GradientTape() as tape_policy:

            loss = loss_function(policy)

            print("expected_reward", -1.0 * loss)

            grad = tape_policy.gradient(
                loss,
                policy.trainable_variables
            )

            grad, sAs = policy.naturalize(
                observations,
                grad
            )

            print("sAs", sAs)

            grad = line_search(
                loss_function,
                policy,
                grad,
                tf.math.sqrt(delta / sAs),
            )

            policy.apply_gradients(grad)




