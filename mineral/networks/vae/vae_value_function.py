"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.vae.vae_network import VAENetwork
from mineral.core.functions.value_function import ValueFunction


class DenseValueFunction(VAENetwork, ValueFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        VAENetwork.__init__(self, *args, **kwargs)

    def get_values(
        self,
        observations,
        **kwargs
    ):
        return self.get_expected_value(observations, **kwargs)
