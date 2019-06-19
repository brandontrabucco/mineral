"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Monitor(ABC):

    @abstractmethod
    def set_step(
        self,
        step
    ):
        return NotImplemented

    @abstractmethod
    def record(
        self,
        key,
        value,
    ):
        return NotImplemented
