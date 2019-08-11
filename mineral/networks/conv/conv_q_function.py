"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.conv.conv_network import ConvNetwork
from mineral.core.functions.q_function import QFunction


class DenseQFunction(ConvNetwork, QFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        ConvNetwork.__init__(self, *args, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions,
        **kwargs
    ):
        return self.get_expected_value(observations, actions, **kwargs)
