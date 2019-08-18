"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(
        self,
        sampler,
        buffers,
        algorithms,
        **kwargs
    ):
        self.sampler = sampler
        self.buffers = buffers if isinstance(buffers, list) else [buffers]
        self.algorithms = algorithms if isinstance(algorithms, list) else [algorithms]

    @abstractmethod
    def train(
        self
    ):
        return NotImplemented
