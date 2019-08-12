"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC
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
        self.model = model

    def get_predictions(
        self,
        observations,
        actions
    ):
        next_observations = self.model.sample(
            observations[:, :(-1), ...],
            actions
        )
        return next_observations
