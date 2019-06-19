"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Base(ABC):

    @abstractmethod
    def gradient_update(
        self, 
        *inputs
    ):
        return NotImplemented