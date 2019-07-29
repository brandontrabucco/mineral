"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.conv.conv_network import ConvNetwork
from mineral.core.functions.forward_model import ForwardModel


class ConvForwardModel(ConvNetwork, ForwardModel):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        ConvNetwork.__init__(self, *args, **kwargs)

    def get_stochastic_observations(
        self,
        observations,
        actions
    ):
        return self.sample(observations, actions)

    def get_deterministic_observations(
        self,
        observations,
        actions
    ):
        return self.get_expected_value(observations, actions)
