"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.networks.dense.dense_network import DenseNetwork
from mineral.core.functions.policy import Policy


class DensePolicy(DenseNetwork, Policy):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        DenseNetwork.__init__(self, *args, **kwargs)

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
