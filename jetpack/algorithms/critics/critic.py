"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.algorithms.base import Base


class Critic(Base, ABC):

    @abstractmethod
    def gradient_update(
        self, 
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def gradient_update_return_weights(
        self,
        *inputs
    ):
        return NotImplemented