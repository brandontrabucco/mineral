"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.latent_variable.latent_variable_network import LatentVariableNetwork
from mineral.core.functions.policy import Policy


class LatentVariablePolicy(LatentVariableNetwork, Policy):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        LatentVariableNetwork.__init__(self, *args, **kwargs)

    def get_stochastic_actions(
        self,
        observations,
        **kwargs
    ):
        return self.sample(observations, **kwargs)

    def get_deterministic_actions(
        self,
        observations,
        **kwargs
    ):
        return self.get_expected_value(observations, **kwargs)
