"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.algorithms.base import Base


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
