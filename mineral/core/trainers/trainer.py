"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(
        self,
        *args
    ):
        self.samplers = args[0::3]
        self.buffers = args[1::3]
        self.algorithms = args[2::3]

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented
