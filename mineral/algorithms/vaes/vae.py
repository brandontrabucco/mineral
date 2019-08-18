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
            **kwargs
        )
        self.master_vae_network = vae_network
        self.worker_vae_network = vae_network.clone()

    def get_encoding(
        self,
        inputs
    ):
        return self.master_vae_network.encoder.get_expected_value(
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
        self.master_vae_network.copy_to(self.worker_vae_network)
        self.update_vae(
            observations,
            actions,
            rewards,
            terminals)
        self.worker_vae_network.copy_to(self.master_vae_network)
