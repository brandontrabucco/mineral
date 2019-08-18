"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as ml
from gym.spaces import Box
from mineral.core.envs.proxy_env import ProxyEnv


class NormalizedEnv(ProxyEnv):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        ProxyEnv.__init__(self, *args, **kwargs)
        self.original_observation_space = self.observation_space.spaces
        self.original_action_space = self.action_space
        self.observation_space = ml.nested_apply(
            create_space,
            self.original_observation_space)
        self.action_space = create_space(self.original_action_space)

    def reset(
        self,
        **kwargs
    ):
        observation = ProxyEnv.reset(self, **kwargs)
        observation = ml.nested_apply(
            normalize,
            observation,
            self.original_observation_space)
        observation = ml.nested_apply(
            lambda x: x.astype(np.float32),
            observation)
        return observation

    def step(
        self, 
        action
    ):
        denormalized_action = denormalize(
            action, self.original_action_space)
        observation, reward, done, info = ProxyEnv.step(
            self, denormalized_action)
        observation = ml.nested_apply(
            normalize, observation, self.original_observation_space)
        observation = ml.nested_apply(
            lambda x: x.astype(np.float32),
            observation)
        return observation, reward, done, info


def create_space(space):
    upper_bound = np.ones(space.shape)
    return Box(-1 * upper_bound, upper_bound)


def denormalize(data, space):
    lower_bound = space.low
    upper_bound = space.high
    if np.any(np.isinf(lower_bound)) or np.any(np.isinf(upper_bound)):
        return data
    return np.clip(
        lower_bound + (data + 1.0) * 0.5 * (
            upper_bound - lower_bound), lower_bound, upper_bound)


def normalize(data, space):
    lower_bound = space.low
    upper_bound = space.high
    if np.any(np.isinf(lower_bound)) or np.any(np.isinf(upper_bound)):
        return data
    return np.clip(
        (data - lower_bound) * 2.0 / (
            upper_bound - lower_bound) - 1.0, -1.0, 1.0)
