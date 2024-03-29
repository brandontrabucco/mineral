"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from mineral.algorithms.base import Base


class VAE(Base, ABC):

    def __init__(
        self,
        vae_network,
        **kwargs
    ):
        Base.__init__(
            self,
            **kwargs)
        self.vae_network = vae_network

    def get_encoding(
        self,
        inputs
    ):
        return self.vae_network.encoder.get_expected_value(
            inputs)

    @abstractmethod
    def update_vae(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        self.update_vae(
            observations,
            actions,
            rewards,
            terminals)
