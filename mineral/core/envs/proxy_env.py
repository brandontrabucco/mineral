"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as ml
from gym import Env
from gym.spaces import Box, Dict
from mineral.core.cloneable import Cloneable


class ProxyEnv(Env, Cloneable):

    def __init__(
        self,
        wrapped_env,
        *wrapped_env_args,
        reward_scale=1.0,
        reward_shift=0.0,
        **wrapped_env_kwargs
    ):
        Cloneable.__init__(
            self,
            wrapped_env,
            *wrapped_env_args,
            reward_scale=reward_scale,
            reward_shift=reward_shift,
            **wrapped_env_kwargs)
        self.wrapped_env = wrapped_env(*wrapped_env_args, **wrapped_env_kwargs)
        self.observation_space = self.wrapped_env.observation_space
        if isinstance(self.observation_space, Box):
            self.observation_space = Dict({
                "proprio_observation": self.observation_space})
        self.action_space = self.wrapped_env.action_space
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

    def copy_to(
        self,
        clone
    ):
        clone.reward_scale = self.reward_scale
        clone.reward_shift = self.reward_shift

    def reset(
        self, 
        **kwargs
    ):
        observation = ml.nested_apply(
            lambda x: np.array(x, dtype=np.float32),
            self.wrapped_env.reset(**kwargs))
        if not isinstance(observation, dict):
            observation = {"proprio_observation": observation}
        return observation

    def step(
        self, 
        action
    ):
        observation, reward, done, info = self.wrapped_env.step(action)
        if not isinstance(observation, dict):
            observation = {"proprio_observation": observation}
        observation = ml.nested_apply(
            lambda x: np.array(x, dtype=np.float32), observation)
        observation = ml.nested_apply(
            lambda x: np.array(x, dtype=np.float32), observation)
        reward = self.reward_shift + self.reward_scale * np.array(
            reward, dtype=np.float32)
        return observation, reward, done, info

    def render(
        self, 
        *args, 
        **kwargs
    ):
        return self.wrapped_env.render(*args, **kwargs)

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
        return '{}({})'.format(type(self).__name__, self.wrapped_env)
