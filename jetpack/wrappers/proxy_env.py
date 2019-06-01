"""Author: Brandon Trabucco, Copyright 2019"""


from gym import Env


class ProxyEnv(Env):

    def __init__(
        self, 
        wrapped_env,
    ):
        self.wrapped_env = wrapped_env
        self.action_space = self.wrapped_env.action_space
        self.observation_space = self.wrapped_env.observation_space

    def reset(
        self, 
        **kwargs,
    ):
        return self.wrapped_env.reset(**kwargs)

    def step(
        self, 
        action,
    ):
        return self.wrapped_env.step(action)

    def render(
        self, 
        *args, 
        **kwargs,
    ):
        return self.wrapped_env.render(
            *args, 
            **kwargs,
        )

    def terminate(
        self,
    ):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(
        self, 
        attr,
    ):
        return getattr(self.wrapped_env, attr)

    def __getstate__(
        self,
    ):
        return self.__dict__

    def __setstate__(
        self, 
        state,
    ):
        self.__dict__.update(state)

    def __str__(
        self,
    ):
        return '{}({})'.format(
            type(self).__name__, 
            self.wrapped_env,
        )