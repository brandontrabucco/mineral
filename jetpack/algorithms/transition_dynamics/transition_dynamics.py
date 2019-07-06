"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.algorithms.base import Base


class TransitionDynamics(Base, ABC):

    @abstractmethod
    def update_transition(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented

    def gradient_update(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        self.update_transition(
            observations,
            actions,
            rewards,
            terminals
        )
