"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.optimizers.kl_constraint import KLConstraint


class GaussianKLConstraint(KLConstraint):

    def get_kl_divergence(
        self,
        mlp_outputs,
        old_mlp_outputs
    ):
        if len(mlp_outputs) > 0:
            mean = mlp_outputs[0]
            std = 1
        if len(mlp_outputs) > 1:
            std = mlp_outputs[1]
        if len(old_mlp_outputs) > 0:
            other_mean = old_mlp_outputs[0]
            other_std = 1
        if len(old_mlp_outputs) > 1:
            other_std = old_mlp_outputs[1]
        std_ratio = tf.square(std / other_std)
        return 0.5 * tf.reduce_sum(
            std_ratio +
            tf.square((other_mean - mean) / other_std) -
            tf.math.log(std_ratio) -
            tf.ones(tf.shape(mean)),
            axis=-1
        )
