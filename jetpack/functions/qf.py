"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class QF(ABC):

    @abstractmethod
    def get_qvalues(
        self,
        observations,
        actions
    ):
        return NotImplemented
