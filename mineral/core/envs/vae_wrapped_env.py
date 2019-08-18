"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from gym.spaces import Box, Dict, Tuple
from mineral.core.envs.proxy_env import ProxyEnv
from mineral.core.cloneable import Cloneable


class VAEWrappedEnv(ProxyEnv, Cloneable):

    def __init__(
        self, 
        wrapped_env,
        vae,
        selector=(lambda x: x),
        assigner=(lambda x, y: y),
        **kwargs
    ):
        ProxyEnv.__init__(self, wrapped_env, **kwargs)
        Cloneable.__init__(
            self,
            wrapped_env,
            vae,
            selector=(lambda x: x),
            assigner=(lambda x, y: y),
            **kwargs)
        observation_space = self.wrapped_env.observation_space
        if (isinstance(observation_space, Dict) or
                isinstance(observation_space, Tuple)):
            observation_space = observation_space.spaces
        self.observation_space = assigner(
            observation_space,
            Box(-1.0 * np.ones([vae.latent_size]), np.ones([vae.latent_size])))
        self.action_space = self.wrapped_env.action_space
        self.vae = vae
        self.selector = selector
        self.assigner = assigner

    def copy_to(
        self,
        clone
    ):
        ProxyEnv.copy_to(clone)
        clone.vae = self.vae
        clone.selector = self.selector
        clone.assigner = self.assigner

    def reset(
        self,
        **kwargs
    ):
        observation = self.selector(ProxyEnv.reset(self, **kwargs))
        encoding = self.vae.encoder.get_expected_value(observation[None, ...])[0]
        return self.assigner(observation, encoding)

    def step(
        self,
        action
    ):
        observation, reward, done, info = ProxyEnv.step(
            self, action)
        encoding = self.vae.encoder.get_expected_value(observation[None, ...])[0]
        return self.assigner(observation, encoding), reward, done, info
