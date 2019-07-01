"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.networks.dense.dense_mlp import DenseMLP
from jetpack.functions.value_function import ValueFunction


class DenseValueFunction(DenseMLP, ValueFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        DenseMLP.__init__(self, *args, **kwargs)

    def get_values(
        self,
        observations
    ):
        return self(observations)
