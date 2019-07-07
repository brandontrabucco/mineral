"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.optimizers.gradients.natural_gradient import NaturalGradient


class GaussianNaturalGradient(NaturalGradient):

    def get_fisher_diagonals(
        self,
        *inputs
    ):
        outputs = []
        if len(inputs) > 0:
            outputs.append(
                tf.ones(tf.shape(inputs[0]))
            )
        if len(inputs) > 1:
            outputs.append(
                2.0 / tf.exp(inputs[1])
            )
        return outputs

