"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class VF(ABC):

    @abstractmethod
    def get_values(
        self,
        observations
    ):
        return NotImplemented