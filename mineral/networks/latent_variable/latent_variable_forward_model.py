"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.latent_variable.latent_variable_network import LatentVariableNetwork
from mineral.core.functions.forward_model import ForwardModel


class LatentVariableForwardModel(LatentVariableNetwork, ForwardModel):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        LatentVariableNetwork.__init__(self, *args, **kwargs)

    def get_stochastic_observations(
        self,
        observations,
        actions,
        **kwargs
    ):
        return self.sample(observations, actions, **kwargs)

    def get_deterministic_observations(
        self,
        observations,
        actions,
        **kwargs
    ):
        return self.get_expected_value(observations, actions, **kwargs)
