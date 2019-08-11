"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.vae.vae_network import VAENetwork
from mineral.core.functions.policy import Policy


class DensePolicy(VAENetwork, Policy):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        VAENetwork.__init__(self, *args, **kwargs)

    def get_stochastic_actions(
        self,
        observations
    ):
        return self.sample(observations)

    def get_deterministic_actions(
        self,
        observations
    ):
        return self.get_expected_value(observations)
