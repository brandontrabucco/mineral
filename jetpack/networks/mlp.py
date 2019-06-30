"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod
from jetpack.fisher import inverse_fisher_vector_product


class MLP(tf.keras.models.Model, ABC):

    def __init__(
        self,
        tau=1e-3,
        naturalize_gradients=False,
        tolerance=1e-3,
        maximum_iterations=100,
        optimizer=tf.keras.optimizers.Adam,
        **optimizer_kwargs
    ):
        tf.keras.models.Model.__init__(self)
        self.tau = tau
        self.naturalize_gradients = naturalize_gradients
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.optimizer = optimizer(**optimizer_kwargs)

    @abstractmethod
    def call(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def fisher_information_matrix(
        self,
        *outputs
    ):
        return NotImplemented

    def inverse_fisher_vector_product(
        self,
        gradients,
        *inputs
    ):
        return inverse_fisher_vector_product(
            lambda: self(*inputs),
            self.fisher_information_matrix,
            self.trainable_variables,
            gradients,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations
        )

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        with tf.GradientTape() as gradient_tape:
            grad = gradient_tape.gradient(
                loss_function(*inputs),
                self.trainable_variables
            )
            if self.naturalize_gradients:
                grad = self.inverse_fisher_vector_product(
                    grad,
                    *inputs
                )
            return grad

    def apply_gradients(
        self,
        gradients
    ):
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_variables
            )
        )

    def soft_update(
        self,
        weights
    ):
        self.set_weights([
            self.tau * w + (1.0 - self.tau) * w_self
            for w, w_self in zip(
                weights,
                self.get_weights()
            )
        ])
