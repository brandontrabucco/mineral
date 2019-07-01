"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.networks.optimizeable import Optimizeable
from jetpack.fisher import inverse_fisher_vector_product


class NaturalGradient(Optimizeable, ABC):

    def __init__(
        self,
        gradient,
        tolerance=1e-3,
        maximum_iterations=100,
        return_sAs=False
    ):
        self.gradient = gradient
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.return_sAs = return_sAs

    @abstractmethod
    def get_hessian_diagonals(
        self,
        *inputs
    ):
        return NotImplemented

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        gradients, sAs = inverse_fisher_vector_product(
            lambda: self.gradient(*inputs),
            self.get_hessian_diagonals,
            self.gradient.trainable_variables,
            self.gradient.compute_gradients(
                loss_function,
                *inputs
            ),
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations
        )
        return (
            gradients, sAs
            if self.return_sAs else gradients
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
