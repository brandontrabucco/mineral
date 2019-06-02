"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
from jetpack.algorithms.base import Base
from jetpack.data.buffer import Buffer


class Trainer(ABC):

    def __init__(
        self,
        buffer: Buffer,
        algorithm: Base
    ):
        self.buffer = buffer
        self.algorithm = algorithm

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented