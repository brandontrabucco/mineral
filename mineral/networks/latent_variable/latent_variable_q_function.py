"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.latent_variable.latent_variable_network import LatentVariableNetwork
from mineral.core.functions.q_function import QFunction


class LatentVariableQFunction(LatentVariableNetwork, QFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        LatentVariableNetwork.__init__(self, *args, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions,
        **kwargs
    ):
        return self.get_expected_value(observations, actions, **kwargs)
