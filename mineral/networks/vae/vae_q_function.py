"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.vae.vae_network import VAENetwork
from mineral.core.functions.q_function import QFunction


class DenseQFunction(VAENetwork, QFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        VAENetwork.__init__(self, *args, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions
    ):
        return self.get_expected_value(observations, actions)
