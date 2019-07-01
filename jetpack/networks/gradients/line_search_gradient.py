"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.mlp import MLP
from jetpack.line_search import line_search


class LineSearchGradient(MLP):

    def __init__(
        self,
        mlp,
        delta=1.0,
        scale_factor=0.5,
        iterations=100,
        use_sAs=False
    ):
        MLP.__init__(self)
        self.mlp = mlp
        self.delta = delta
        self.scale_factor = scale_factor
        self.iterations = iterations
        self.use_sAs = use_sAs

    def call(
        self,
        *inputs
    ):
        return self.mlp(*inputs)

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

    def apply_gradients(
        self,
        gradients
    ):
        self.mlp.apply_gradients(
            gradients
        )

    def soft_update(
        self,
        weights
    ):
        self.mlp.soft_update(
            weights
        )

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.mlp, attr)
