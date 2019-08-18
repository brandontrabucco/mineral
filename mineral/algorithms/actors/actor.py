"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from mineral.algorithms.base import Base


class Actor(Base, ABC):

    @abstractmethod
    def update_actor(
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
        self.update_actor(
            observations,
            actions,
            rewards,
            terminals)
