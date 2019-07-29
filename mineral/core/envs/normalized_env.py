"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as jp
from gym.spaces import Box, Dict, Tuple
from mineral.core.envs.proxy_env import ProxyEnv


class NormalizedEnv(ProxyEnv):

    def __init__(
        self, 
        wrapped_env,
        **kwargs
    ):
        ProxyEnv.__init__(self, wrapped_env, **kwargs)
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        self.observation_space = jp.nested_apply(
            create_space,
            observation_space)
        self.action_space = create_space(self.wrapped_env.action_space)

    def reset(
        self,
        **kwargs
    ):
        observation = ProxyEnv.reset(self, **kwargs)
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        observation = jp.nested_apply(
            normalize,
            observation,
            observation_space)
        return observation

    def step(
        self, 
        action
    ):
        observation, reward, done, info = ProxyEnv.step(
            self,
            denormalize(action, self.wrapped_env.action_space))
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        observation = jp.nested_apply(
            normalize,
            observation,
            observation_space)
        return observation, reward, done, info


def create_space(space):
    upper_bound = np.ones(space.shape)
    return Box(-1 * upper_bound, upper_bound)


def denormalize(data, space):
    lower_bound = space.low
    upper_bound = space.high
    return np.clip(
        lower_bound + (data + 1.0) * 0.5 * (upper_bound - lower_bound),
        lower_bound,
        upper_bound)


def normalize(scaled_data, space):
    lower_bound = space.low
    upper_bound = space.high
    return np.clip(
        (scaled_data - lower_bound) * 2.0 / (upper_bound - lower_bound) - 1.0,
        -1.0,
        1.0)
