"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from gym.spaces import Box, Dict, Tuple
from jetpack.wrappers.proxy_env import ProxyEnv


class NormalizedEnv(ProxyEnv):

    def __init__(
        self, 
        wrapped_env,
        reward_scale=1.0
    ):
        def create_space(space):
            upper_bound = np.ones(space.shape)
            return Box(-1 * upper_bound, upper_bound)
        ProxyEnv.__init__(
            self,
            wrapped_env,
            reward_scale=reward_scale
        )
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        self.observation_space = jp.nested_apply(
            create_space,
            observation_space
        )
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
            observation_space
        )
        return observation

    def step(
        self, 
        action
    ):
        observation, reward, done, info = ProxyEnv.step(
            self,
            denormalize(action, self.wrapped_env.action_space)
        )
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        observation = jp.nested_apply(
            normalize,
            observation,
            observation_space
        )
        return observation, reward, done, info


def denormalize(data, space):
    lower_bound = space.low
    upper_bound = space.high
    scaled_data = np.clip(
        lower_bound + (data + 1.0) * 0.5 * (upper_bound - lower_bound),
        lower_bound,
        upper_bound
    )
    return scaled_data


def normalize(scaled_data, space):
    lower_bound = space.low
    upper_bound = space.high
    data = np.clip(
        (scaled_data - lower_bound) * 2.0 / (upper_bound - lower_bound) - 1.0,
        -1.0,
        1.0
    )
    return data
