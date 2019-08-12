"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(
        self,
        *inputs
    ):
        self.buffers = inputs[0::2]
        self.algorithms = inputs[1::2]

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented