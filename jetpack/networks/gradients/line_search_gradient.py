"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.gradient import Gradient
from jetpack.line_search import line_search


class LineSearchGradient(Gradient):

    def __init__(
        self,
        gradient,
        delta=1.0,
        scale_factor=0.5,
        iterations=100,
        use_sAs=False
    ):
        self.gradient = gradient
        self.delta = delta
        self.scale_factor = scale_factor
        self.iterations = iterations
        self.use_sAs = use_sAs

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        outputs = self.gradient.compute_gradients(
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
            self.gradient,
            gradients,
            alpha,
            scale_factor=self.scale_factor,
            iterations=self.iterations
        )

    def apply_gradients(
        self,
        gradients
    ):
        self.gradient.apply_gradients(
            gradients
        )

    def __call__(
        self,
        *inputs
    ):
        return self.gradient(*inputs)

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.gradient, attr)
