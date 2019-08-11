"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.conv.conv_network import ConvNetwork
from mineral.core.functions.value_function import ValueFunction


class ConvValueFunction(ConvNetwork, ValueFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        ConvNetwork.__init__(self, *args, **kwargs)

    def get_values(
        self,
        observations,
        **kwargs
    ):
        return self.get_expected_value(observations, **kwargs)
