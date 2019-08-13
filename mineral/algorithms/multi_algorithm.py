"""Author: Brandon Trabucco, Copyright 2019"""


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
        for alg in self.algorithms:
            alg.gradient_update(
                observations,
                actions,
                rewards,
                terminals)
