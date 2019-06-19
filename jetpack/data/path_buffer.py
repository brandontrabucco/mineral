"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from abc import ABC, abstractmethod
from jetpack.data.buffer import Buffer
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.functions.policy import Policy


class PathBuffer(Buffer, ABC):

    def __init__(
        self,
        env: ProxyEnv,
        policy: Policy,
        selector=None,
    ):
        self.selector = (lambda x: x) if selector is None else selector
        Buffer.__init__(
            self, 
            env,
            policy
        )

    def reset(
        self,
        max_size,
        max_path_length
    ):
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.size = 0
        self.head = 0
        self.tail = np.zeros(
            [self.max_size],
            dtype=np.int32
        )
        self.seen = np.zeros(
            [self.max_size, self.max_path_length],
            dtype=np.int32
        )
        self.candidates = np.zeros(
            [0, 2],
            dtype=np.int32
        )

    def explore(
        self,
        num_paths_to_collect,
        render,
        render_kwargs
    ):
        exploration_returns = []
        for i in range(num_paths_to_collect):
            observation = self.env.reset()
            path_return = 0.0
            for j in range(self.max_path_length):
                if j > 0 and self.seen[self.head, j - 1] == 0:
                    self.candidates = np.concatenate([
                        self.candidates,
                        np.array([[self.head, j - 1]])
                    ], 0)
                    self.seen[self.head, j - 1] = 1
                action = self.policy.get_stochastic_actions(
                    self.selector(observation)[np.newaxis, ...]
                ).numpy()[0, ...]
                next_observation, reward, done, info = self.env.step(
                    action
                )
                if render:
                    self.env.render(**render_kwargs)
                if self.size == 0:
                    def create(x): 
                        return np.zeros([
                            self.max_size, 
                            self.max_path_length,
                            *x.shape
                        ])
                    self.observations = jp.nested_apply(
                        create,
                        observation
                    )
                    self.actions = jp.nested_apply(
                        create,
                        action
                    )
                    self.rewards = jp.nested_apply(
                        create,
                        reward
                    )
                def put(x, y):
                    x[self.head, j, ...] = y
                jp.nested_apply(
                    put,
                    self.observations,
                    observation
                )
                jp.nested_apply(
                    put,
                    self.actions,
                    action
                )
                jp.nested_apply(
                    put,
                    self.rewards,
                    reward
                )
                path_return = path_return + reward
                self.tail[self.head] = j
                observation = next_observation
                if done:
                    break
            self.head = (self.head + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
            exploration_returns.append(path_return)
        return np.mean(exploration_returns)

    def evaluate(
        self,
        num_paths_to_collect,
        render,
        render_kwargs
    ):
        evaluation_returns = []
        for i in range(num_paths_to_collect):
            observation = self.env.reset()
            path_return = 0.0
            for i in range(self.max_path_length):
                action = self.policy.get_deterministic_actions(
                    self.selector(observation)[np.newaxis, ...]
                ).numpy()[0, ...]
                next_observation, reward, done, info = self.env.step(
                    action
                )
                if render:
                    self.env.render(**render_kwargs)
                path_return = path_return + reward
                observation = next_observation
                if done:
                    break
            evaluation_returns.append(path_return)
        return np.mean(evaluation_returns)

    @abstractmethod
    def sample(
        self,
        batch_size
    ):
        return NotImplemented
