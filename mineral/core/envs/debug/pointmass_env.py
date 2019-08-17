"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from gym.spaces import Box, Dict
from gym import Env


class PointmassEnv(Env):

    def __init__(
        self, size=2, ord=2
    ):
        self.observation_space = Dict({
            "proprio_observation": Box(
                -1.0 * np.ones([size]), np.ones([size]))})
        self.action_space = Box(
            -1.0 * np.ones([size]), np.ones([size]))
        self.position = np.zeros([size])
        self.goal = np.ones([size])
        self.size = size
        self.ord = ord

    def reset(
        self,
        **kwargs
    ):
        self.position = np.zeros([self.size])
        return {"proprio_observation": self.position}

    def step(
        self, 
        action
    ):
        self.position = np.clip(
            self.position + np.clip(
                action, -1.0 * np.ones([self.size]),
                np.ones([self.size])),
            -1.0 * np.ones([self.size]), np.ones([self.size]))
        return {"proprio_observation": self.position}, -1.0 * np.linalg.norm(
            self.position - self.goal, ord=self.ord), False, {}

    def render(
        self,
        image_size=256,
        **kwargs
    ):
        image = np.zeros([image_size, image_size, 3])
        x, y = np.meshgrid(np.arange(image_size),
                           np.arange(image_size))
        goal_radius = np.sqrt((x - (self.goal[0] + 1.0) * image_size / 2)**2 + (
                y - (self.goal[1] + 1.0) * image_size / 2)**2)
        position_radius = np.sqrt((x - (self.position[0] + 1.0) * image_size / 2)**2 + (
                y - (self.position[1] + 1.0) * image_size / 2)**2)
        image[:, :, 1] = np.ones(goal_radius.shape) / (
                1.0 + goal_radius / image_size * 12.0)
        image[:, :, 2] = np.ones(position_radius.shape) / (
                1.0 + position_radius / image_size * 12.0)
        return image
