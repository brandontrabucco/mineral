"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class ValueFunction(ABC):

    @abstractmethod
    def get_values(
        self,
        observations,
        **kwargs
    ):
        return NotImplemented
