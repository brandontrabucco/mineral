"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.optimizers.optimizer import Optimizer
from mineral.optimizers.utils.fisher import inverse_fisher_vector_product


class NaturalGradient(Optimizer):

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

    def compute_gradients(
        self,
        loss_function,
        *inputs,
        **kwargs
    ):
        gradients, sAs = inverse_fisher_vector_product(
            lambda: self.mlp.get_parameters(*inputs),
            lambda: self.mlp.get_fisher_information(*inputs),
            self.mlp.trainable_variables,
            self.mlp.compute_gradients(loss_function, *inputs, **kwargs),
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations)
        return (gradients, sAs) if self.return_sAs else gradients
