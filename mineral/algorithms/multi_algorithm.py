"""Author: Brandon Trabucco, Copyright 2019"""


import threading
from mineral.algorithms.base import Base


class MultiAlgorithm(Base):

    def __init__(
        self,
        *algorithms,
        **kwargs
    ):
        Base.__init__(self, **kwargs)
        self.algorithms = algorithms

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        pass

    def gradient_update(
        self,
        buffer
    ):
        self.iteration += 1
        if (self.iteration >= self.update_after) and (
                self.iteration - self.last_update_iteration >= self.update_every):
            self.last_update_iteration = self.iteration

            def inner_gradient_update(algorithm, inner_buffer):
                algorithm.gradient_update(inner_buffer)

            threads = [threading.Thread(
                target=inner_gradient_update, args=(a, buffer))
                for a in self.algorithms]

            for t in threads:
                t.start()
            for t in threads:
                t.join()
