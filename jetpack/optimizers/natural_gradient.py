"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.optimizers.optimizer import Optimizer
from jetpack.optimizers.utils.fisher import inverse_fisher_vector_product


class NaturalGradient(Optimizer, ABC):

    def __init__(
        self,
        mlp,
        tolerance=1e-3,
        maximum_iterations=100,
        return_sAs=False
    ):
        Optimizer.__init__(self, mlp)
        self.tolerance = tolerance
        self.maximum_iterations = maximum_iterations
        self.return_sAs = return_sAs

    @abstractmethod
    def get_fisher_diagonals(
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
            self.get_fisher_diagonals,
            self.mlp.trainable_variables,
            self.mlp.compute_gradients(
                loss_function,
                *inputs
            ),
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations
        )
        return (
            (gradients, sAs)
            if self.return_sAs else gradients
        )
