"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Base(ABC):

    def __init__(
        self,
        selector=None,
        monitor=None,
    ):
        self.selector = (lambda x: x) if selector is None else selector
        self.monitor = monitor
        self.iteration = 0

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
        observations,
        actions,
        rewards,
        terminals
    ):
        if self.monitor is not None:
            self.monitor.set_step(self.iteration)
        self.iteration += 1
        self.update_algorithm(
            self.selector(observations),
            actions,
            rewards,
            terminals
        )
