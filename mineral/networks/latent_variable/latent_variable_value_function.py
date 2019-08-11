"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.latent_variable.latent_variable_network import LatentVariableNetwork
from mineral.core.functions.value_function import ValueFunction


class LatentVariableValueFunction(LatentVariableNetwork, ValueFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        LatentVariableNetwork.__init__(self, *args, **kwargs)

    def get_values(
        self,
        observations,
        **kwargs
    ):
        return self.get_expected_value(observations, **kwargs)
