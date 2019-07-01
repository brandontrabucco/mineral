"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Gradient(ABC):

    @abstractmethod
    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def apply_gradients(
        self,
        gradients
    ):
        return NotImplemented

    def minimize(
        self,
        loss_function,
        *inputs
    ):
        self.apply_gradients(
            self.compute_gradients(
                loss_function,
                *inputs
            )
        )
