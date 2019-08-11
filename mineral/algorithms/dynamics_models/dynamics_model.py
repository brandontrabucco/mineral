"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from mineral.algorithms.base import Base


class DynamicsModel(Base, ABC):

    @abstractmethod
    def get_predictions(
        self,
        observations,
        actions
    ):
        return NotImplemented

    @abstractmethod
    def update_model(
        self,
        observations,
        actions,
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
        Base.gradient_update(
            self,
            observations,
            actions,
            rewards,
            terminals
        )
        self.update_model(
            observations,
            actions,
            terminals
        )
