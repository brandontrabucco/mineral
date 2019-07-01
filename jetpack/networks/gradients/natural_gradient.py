"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.networks.mlp import MLP
from jetpack.fisher import inverse_fisher_vector_product


class NaturalGradient(MLP, ABC):

    def __init__(
        self,
        mlp,
        tolerance=1e-3,
        maximum_iterations=100,
        return_sAs=False
    ):
        MLP.__init__(self)
        self.mlp = mlp
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.return_sAs = return_sAs

    def call(
        self,
        *inputs
    ):
        return self.mlp(*inputs)

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
            lambda: self.mlp(*inputs),
            self.get_hessian_diagonals,
            self.mlp.trainable_variables,
            self.mlp.compute_gradients(
                loss_function,
                *inputs
            ),
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations
        )
        return (
            gradients, sAs
            if self.return_sA else gradients
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
