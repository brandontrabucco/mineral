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

    def reset(
        self,
        **kwargs
    ):
        observation = PointmassEnv.reset(self, **kwargs)
        return {"image_observation": self.render(mode='rgb_array', image_size=self.image_size),
                **observation}

    def step(
        self, 
        action
    ):
        observation, reward, done, info = PointmassEnv.step(self, action)
        return {"image_observation": self.render(mode='rgb_array', image_size=self.image_size),
                **observation}, reward, done, info
