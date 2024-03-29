"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.core.has_gradient import HasGradient


class Optimizer(HasGradient):

    def __init__(
        self,
        mlp,
        **kwargs
    ):
        self.mlp = mlp

    def compute_gradients(
        self,
        loss_function,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    def apply_gradients(
        self,
        gradients
    ):
        self.mlp.apply_gradients(
            gradients
        )

    def __call__(
        self,
        *inputs
    ):
        return self.mlp(*inputs)

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.mlp, attr)
