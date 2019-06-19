"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.vf import VF


class DenseVF(DenseMLP, VF):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)

    def get_values(
        self,
        observations
    ):
        return self(observations)
