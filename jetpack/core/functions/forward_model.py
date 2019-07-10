"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class ForwardModel(ABC):

    @abstractmethod
    def get_stochastic_observations(
        self,
        observations,
        actions
    ):
        return NotImplemented

    @abstractmethod
    def get_deterministic_observations(
        self,
        observations,
        actions
    ):
        return NotImplemented
