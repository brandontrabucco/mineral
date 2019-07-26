"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from gym.spaces import Box
from gym import Env


class PointmassEnv(Env):

    def __init__(
        self, size=2, ord=2
    ):
        self.observation_space = Box(
            -1.0 * np.ones([size]), np.ones([size]))
        self.action_space = Box(
            -1.0 * np.ones([size]), np.ones([size]))
        self.position = np.random.normal(0.0, 0.1, [size])
        self.goal = np.ones([size])
        self.size = size
        self.ord = ord

    def reset(
        self,
        **kwargs
    ):
        self.position = np.random.normal(0.0, 0.1, [self.size])

    def step(
        self, 
        action
    ):
        self.position = np.clip(
            self.position + np.clip(action, -1.0 * np.ones([self.size]), np.ones([self.size])),
            -1.0 * np.ones([self.size]), np.ones([self.size]))
        return self.position, -1.0 * np.linalg.norm(
            self.position - self.goal, ord=self.ord), False, {}
