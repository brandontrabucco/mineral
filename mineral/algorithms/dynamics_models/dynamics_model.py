"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC,abstractmethod
from mineral.algorithms.base import Base


class DynamicsModel(Base, ABC):

    def __init__(
        self,
        model,
        **kwargs
    ):
        Base.__init__(
            self,
            **kwargs
        )
        self.master_model = model
        self.worker_model = model.clone()

    def get_predictions(
        self,
        observations,
        actions
    ):
        next_observations = self.master_model.sample(
            observations[:, :(-1), ...],
            actions)
        return next_observations

    @abstractmethod
    def update_model(
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
        self.master_model.copy_to(self.worker_model)
        self.update_model(
            observations,
            actions,
            rewards,
            terminals)
        self.worker_model.copy_to(self.master_model)
