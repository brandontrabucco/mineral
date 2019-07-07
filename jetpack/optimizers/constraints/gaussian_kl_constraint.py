"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.optimizers.constraints.kl_constraint import KLConstraint


class GaussianKLConstraint(KLConstraint):

    def get_kl_divergence(
        self,
        mlp_outputs,
        old_mlp_outputs
    ):
        mean = mlp_outputs[0]
        log_variance = tf.ones(tf.shape(mean))
        if len(mlp_outputs) > 1:
            log_variance = mlp_outputs[1]
        other_mean = old_mlp_outputs[0]
        other_log_variance = tf.ones(tf.shape(other_mean))
        if len(old_mlp_outputs) > 1:
            other_log_variance = old_mlp_outputs[1]
        return 0.5 * tf.reduce_sum(
            tf.math.exp(log_variance - other_log_variance) +
            other_log_variance - log_variance -
            tf.square(other_mean - mean) / tf.math.exp(
                other_log_variance) +
            tf.ones(tf.shape(mean)), axis=-1)
