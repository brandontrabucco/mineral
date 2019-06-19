"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.dense_mlp import DenseMLP
from jetpack.functions.qf import QF


class DenseQF(DenseMLP, QF):

    def __init__(
        self,
        hidden_sizes,
        **kwargs
    ):
        DenseMLP.__init__(self, hidden_sizes, **kwargs)

    def get_qvalues(
        self,
        observations,
        actions
    ):
        return self(
            tf.concat([
                observations,
                actions
            ], 1)
        )
