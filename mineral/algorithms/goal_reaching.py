"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.base import Base


class GoalReaching(Base):

    def __init__(
        self,
        algorithm,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        order=2,
        **kwargs
    ):
        Base.__init__(self, **kwargs)
        self.algorithm = algorithm
        self.observation_selector = observation_selector
        self.goal_selector = goal_selector
        self.order = order

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        error = (self.observation_selector(observations) -
                 self.goal_selector(observations))
        rewards = tf.linalg.norm(error, ord=self.order, axis=(-1))
        return self.algorithm.update_algorithm(
            observations,
            actions,
            rewards,
            terminals)
