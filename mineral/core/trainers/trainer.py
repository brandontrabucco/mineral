"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(
        self,
        sampler,
        *args
    ):
        self.sampler = sampler
        self.buffers = args[0::2]
        self.algorithms = args[1::2]

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented
