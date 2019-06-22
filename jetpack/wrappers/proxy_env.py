"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from gym import Env
from gym.spaces import Box


class ProxyEnv(Env):

    def __init__(
        self, 
        wrapped_env,
        reward_scale=1.0
    ):
        self.wrapped_env = wrapped_env
        self.observation_space = self.wrapped_env.observation_space
        ub = np.ones(self.wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.reward_scale = reward_scale

    def reset(
        self, 
        **kwargs
    ):
        return jp.nested_apply(
            (lambda x: np.array(x, dtype=np.float32)),
            self.wrapped_env.reset(**kwargs)
        )

    def step(
        self, 
        action
    ):
        lower_bound = self.wrapped_env.action_space.low
        upper_bound = self.wrapped_env.action_space.high
        scaled_action = np.clip(
            lower_bound + (action + 1.0) * 0.5 * (upper_bound - lower_bound),
            lower_bound,
            upper_bound
        )
        observation, reward, done, info = self.wrapped_env.step(
            scaled_action
        )
        observation = jp.nested_apply(
            (lambda x: np.array(x, dtype=np.float32)),
            observation
        )
        reward = self.reward_scale * np.array(
            reward,
            dtype=np.float32
        )
        return observation, reward, done, info

    def render(
        self, 
        *args, 
        **kwargs
    ):
        return self.wrapped_env.render(
            *args, 
            **kwargs
        )

    def terminate(
        self
    ):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

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
            self.wrapped_env
        )