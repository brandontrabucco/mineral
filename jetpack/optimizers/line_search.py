"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.optimizers.optimizer import Optimizer
from jetpack.utils.line_search import line_search


class LineSearch(Optimizer):

    def __init__(
        self,
        mlp,
        delta=1.0,
        scale_factor=0.5,
        iterations=100,
        use_sAs=False
    ):
        Optimizer.__init__(self, mlp)
        self.delta = delta
        self.scale_factor = scale_factor
        self.iterations = iterations
        self.use_sAs = use_sAs

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        outputs = self.mlp.compute_gradients(
            loss_function,
            *inputs
        )

        if self.use_sAs:
            gradients, sAs = outputs
            alpha = tf.math.sqrt(self.delta / sAs)
        else:
            gradients = outputs
            alpha = self.delta

        return line_search(
            loss_function,
            self.mlp,
            gradients,
            alpha,
            scale_factor=self.scale_factor,
            iterations=self.iterations
        )
