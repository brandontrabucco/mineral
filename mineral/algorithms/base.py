"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Base(ABC):

    def __init__(
        self,
        update_every=1,
        update_after=1,
        batch_size=32,
        selector=None,
        monitor=None,
    ):
        self.update_every = update_every
        self.update_after = update_after
        self.batch_size = batch_size
        self.selector = (lambda x: x) if selector is None else selector
        self.monitor = monitor
        self.iteration = 0
        self.last_update_iteration = 0

    @abstractmethod
    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented

    def gradient_update(
        self,
        buffer
    ):
        self.iteration += 1
        if (self.iteration >= self.update_after) and (
                self.iteration - self.last_update_iteration >= self.update_every):
            self.last_update_iteration = self.iteration
            (observations,
                actions,
                rewards,
                terminals) = buffer.sample(self.batch_size)
            self.update_algorithm(
                self.selector(observations),
                actions,
                rewards,
                terminals)
