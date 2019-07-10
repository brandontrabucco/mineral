"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def get_stochastic_actions(
        self,
        observations
    ):
        return NotImplemented

    @abstractmethod
    def get_deterministic_actions(
        self,
        observations
    ):
        return NotImplemented
