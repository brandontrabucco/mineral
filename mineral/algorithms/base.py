"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC


class Base(ABC):

    def __init__(
        self,
        monitor=None
    ):
        self.monitor = monitor
        self.iteration = 0

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
