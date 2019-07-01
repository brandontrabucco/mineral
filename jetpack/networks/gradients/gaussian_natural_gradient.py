"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.gradients.natural_gradient import NaturalGradient


class GaussianNaturalGradient(NaturalGradient):

    def get_hessian_diagonals(
        self,
        *inputs
    ):
        outputs = [
            tf.ones(tf.shape(inputs[0]))
        ]
        if len(inputs) == 2:
            outputs.append(
                2.0 / tf.square(inputs[1])
            )
        return outputs

