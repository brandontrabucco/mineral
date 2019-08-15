"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.core.envs.debug.pointmass_env import PointmassEnv
from gym.spaces import Box


class ImagePointmassEnv(PointmassEnv):

    def __init__(
        self, image_size=48, **kwargs
    ):
        PointmassEnv.__init__(self, **kwargs)
        self.image_size = image_size
        self.observation_space = {**self.observation_space, "image_observation": Box(
            np.zeros([image_size, image_size, 3]), np.ones([image_size, image_size, 3]))}

    def get_image(self):
        image = np.zeros([self.image_size, self.image_size, 3])
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        goal_radius = np.sqrt((x - (self.goal[0] + 1.0) * self.image_size / 2)**2 + (
                y - (self.goal[1] + 1.0) * self.image_size / 2)**2)
        position_radius = np.sqrt((x - (self.position[0] + 1.0) * self.image_size / 2)**2 + (
                y - (self.position[1] + 1.0) * self.image_size / 2)**2)
        image[:, :, 1] = np.ones(goal_radius.shape) / (1.0 + goal_radius / self.image_size * 12.0)
        image[:, :, 2] = np.ones(position_radius.shape) / (1.0 + position_radius / self.image_size * 12.0)
        return image

    def reset(
        self,
        **kwargs
    ):
        observation = PointmassEnv.reset(self, **kwargs)
        return {"image_observation": self.get_image(),
                **observation}

    def step(
        self, 
        action
    ):
        observation, reward, done, info = PointmassEnv.step(self, action)
        return {"image_observation": self.get_image(),
                **observation}, reward, done, info