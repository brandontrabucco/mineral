"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.networks.policies.tanh_gaussian_policy import TanhGaussianPolicy
from jetpack.networks.policies.mean_policy import MeanPolicy
from jetpack.functions.policy import Policy


class TanhMeanPolicy(TanhGaussianPolicy, Policy):

    def __init__(
        self,
        hidden_sizes,
        sigma=1.0,
        **kwargs
    ):
        TanhGaussianPolicy.__init__(self, hidden_sizes, **kwargs)
        self.sigma = sigma

    def get_mean_std(
        self,
        observations
    ):
        return MeanPolicy.get_mean_std(
            self,
            observations
        )

    def fisher_vector_product(
        self,
        observations,
        y
    ):
        return MeanPolicy.fisher_vector_product(
            self,
            observations,
            y
        )
