"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as jp
from gym import Env
from gym.spaces import Box, Dict


class ProxyEnv(Env):

    def __init__(
        self, 
        wrapped_env,
        reward_scale=1.0,
        reward_shift=0.0
    ):
        self.wrapped_env = wrapped_env
        self.observation_space = self.wrapped_env.observation_space
        if isinstance(self.observation_space, Box):
            self.observation_space = Dict({
                "proprio_observation": self.observation_space})
        self.action_space = self.wrapped_env.action_space
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

    def reset(
        self, 
        **kwargs
    ):
        observation = jp.nested_apply(
            (lambda x: np.array(x, dtype=np.float32)),
            self.wrapped_env.reset(**kwargs))
        if not isinstance(observation, dict):
            observation = {"proprio_observation": observation}
        return observation

    def step(
        self, 
        action
    ):
        observation, reward, done, info = self.wrapped_env.step(
            action)
        if not isinstance(observation, dict):
            observation = {"proprio_observation": observation}
        observation = jp.nested_apply(
            lambda x: np.array(x, dtype=np.float32),
            observation)
        observation = jp.nested_apply(
            lambda x: np.array(x, dtype=np.float32),
            observation)
        reward = self.reward_shift + self.reward_scale * np.array(
            reward,
            dtype=np.float32)
        return observation, reward, done, info

    def render(
        self, 
        *args, 
        **kwargs
    ):
        return self.wrapped_env.render(
            *args, 
            **kwargs)

    def __getattr__(
        self, 
        attr
    ):
        return getattr(self.wrapped_env, attr)

    def __getstate__(
        self
    ):
        return self.__dict__

    def __setstate__(
        self, 
        state
    ):
        self.__dict__.update(state)

    def __str__(
        self
    ):
        return '{}({})'.format(
            type(self).__name__, 
            self.wrapped_env)
