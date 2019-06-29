"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.functions.policy import Policy


class MeanPolicy(GaussianPolicy, Policy):

    def __init__(
        self,
        hidden_sizes,
        sigma=1.0,
        **kwargs
    ):
        GaussianPolicy.__init__(self, hidden_sizes, **kwargs)
        self.sigma = sigma

    def get_mean_std(
        self,
        observations
    ):
        mean = self(observations)
        return mean, tf.fill(tf.shape(mean), self.sigma)

    def fisher_vector_product(
        self,
        observations,
        y
    ):
        with tf.GradientTape(persistent=True) as tape_policy:
            mean, std = self.get_mean_std(observations)
            mean_v = tf.ones(tf.shape(mean))
            tape_policy.watch(mean_v)
            mean_g = tape_policy.gradient(
                mean,
                self.trainable_variables,
                output_gradients=mean_v
            )
        mean_jvp = tape_policy.gradient(
            mean_g,
            mean_v,
            output_gradients=y
        )
        mean_fvp = tape_policy.gradient(
            mean,
            self.trainable_variables,
            output_gradients=mean_jvp
        )
        return mean_fvp