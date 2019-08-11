"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.dense.dense_network import DenseNetwork
from mineral.core.functions.q_function import QFunction


class DenseQFunction(DenseNetwork, QFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        DenseNetwork.__init__(self, *args, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions,
        **kwargs
    ):
        return self.get_expected_value(observations, actions, **kwargs)
