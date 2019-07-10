"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(
        self,
        buffer,
        algorithm
    ):
        self.buffer = buffer
        self.algorithm = algorithm

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented