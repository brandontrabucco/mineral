"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.value_function import ValueFunction


class DenseValueFunction(DenseMLP, ValueFunction):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)

    def get_values(
        self,
        observations
    ):
        return self(observations)
