"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.optimizers.optimizer import Optimizer


class KLPenalty(Optimizer):

    def __init__(
        self,
        mlp,
        other_mlp,
        iterations_per_copy=1,
        alpha=1.0
    ):
        Optimizer.__init__(self, mlp)
        self.other_mlp = other_mlp
        self.iterations_per_copy = iterations_per_copy
        self.alpha = alpha
        self.iteration = 0

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        def wrapped_loss_function():
            return loss_function() + self.alpha * tf.reduce_mean(
                self.mlp.get_kl_divergence(
                    self.other_mlp, *inputs))
        return self.mlp.compute_gradients(
            wrapped_loss_function, *inputs)

    def apply_gradients(
        self,
        gradients
    ):
        self.mlp.apply_gradients(gradients)
        self.iteration += 1
        if (self.iterations_per_copy is not None and
                self.iteration > self.iterations_per_copy):
            self.other_mlp.set_weights(self.mlp.get_weights())
            self.iteration = 0
