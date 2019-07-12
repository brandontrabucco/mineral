"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.dense.dense_network import DenseNetwork
from mineral.core.functions.value_function import ValueFunction


class DenseValueFunction(DenseNetwork, ValueFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        DenseNetwork.__init__(self, *args, **kwargs)

    def get_values(
        self,
        observations
    ):
        return self.get_expected_value(observations)
