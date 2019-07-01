"""Author: Brandon Trabucco, Copyright 2019"""

from jetpack.networks.dense.dense_mlp import DenseMLP
from jetpack.functions.q_function import QFunction


class DenseQFunction(DenseMLP, QFunction):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        DenseMLP.__init__(self, *args, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions
    ):
        return self(
            observations,
            actions
        )