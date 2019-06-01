"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from jetpack.data.buffer import Buffer
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.core.policy import Policy


class ExperienceReplay(Buffer):

    def __init__(
        self,
        selector,
        env: ProxyEnv,
        policy: Policy,
    ):
        ExperienceReplay.__init__(
            self, 
            env,
            policy,
        )
        self.selector = selector

    def reset(
        self,
        max_size,
    ):
        self.max_size = max_size
        self.size = 0
        self.head = 0

    def collect(
        self,
        num_paths_to_collect,
        max_path_length,
    ):
        num_paths_collected = 0
        while num_paths_collected < num_paths_to_collect:
            observation = self.env.reset()
            for i in range(max_path_length):
                selected_observation = self.selector(observation)
                action = self.policy.get_stochastic_actions(
                    selected_observation[np.newaxis, ...],
                ).numpy()[0, ...]
                next_observation, reward, done, info = self.env.step(
                    action,
                )
                if self.size == 0:
                    def create(x): 
                        return np.zeros([
                            self.max_size, 
                            *x.shape,
                        ])
                    self.observations = jp.nested_apply(
                        create,
                        observation,
                    )
                    self.actions = jp.nested_apply(
                        create,
                        action,
                    )
                    self.rewards = jp.nested_apply(
                        create,
                        reward,
                    )
                    self.next_observations = jp.nested_apply(
                        create,
                        next_observation,
                    )
                def put(x, y):
                    x[self.head, ...] = y
                jp.nested_apply(
                    put,
                    self.observations,
                    observation,
                )
                jp.nested_apply(
                    put,
                    self.actions,
                    action,
                )
                jp.nested_apply(
                    put,
                    self.rewards,
                    reward,
                )
                jp.nested_apply(
                    put,
                    self.next_observations,
                    next_observation,
                )
                self.head = (self.head + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
                if done:
                    break
            num_paths_collected += 1
                

    def sample(
        self,
        batch_size,
    ):
        indices = np.random.choice(
            self.size, 
            size=batch_size, 
            replace=(self.size < batch_size),
        )
        select = lambda x: x[indices, ...]
        return (
            jp.nested_apply(
                select,
                self.observations,
            ),
            jp.nested_apply(
                select,
                self.actions,
            ),
            jp.nested_apply(
                select,
                self.rewards,
            ),
            jp.nested_apply(
                select,
                self.next_observations,
            ),
        )
