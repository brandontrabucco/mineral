"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from collections import OrderedDict
from jetpack.data.experience_replay import ExperienceReplay
from jetpack.networks.policy import Policy


class SimpleExperienceReplay(ExperienceReplay):

    def __init__(
        self,
        env,
        policy: Policy,
    ):
        ExperienceReplay.__init__(
            self, 
            env,
            policy,
        )

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
                action = self.policy.get_stochastic_actions(
                    observation,
                ).numpy()
                next_observation, reward, done, info = self.env.step(
                    action,
                )
                if self.size == 0:
                    create = lambda x: np.zeros([
                        self.max_size, 
                        *x.shape[1:],
                    ])
                    self.observations = ExperienceReplay.nested_apply(
                        create,
                        observation,
                    )
                    self.actions = ExperienceReplay.nested_apply(
                        create,
                        action,
                    )
                    self.rewards = ExperienceReplay.nested_apply(
                        create,
                        reward,
                    )
                    self.next_observations = ExperienceReplay.nested_apply(
                        create,
                        next_observation,
                    )
                def put(x, y):
                    x[self.head, ...] = y[0, ...]
                ExperienceReplay.nested_apply(
                    put,
                    self.observations,
                    observation,
                )
                ExperienceReplay.nested_apply(
                    put,
                    self.actions,
                    action,
                )
                ExperienceReplay.nested_apply(
                    put,
                    self.rewards,
                    reward,
                )
                ExperienceReplay.nested_apply(
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
            ExperienceReplay.nested_apply(
                select,
                self.observations,
            ),
            ExperienceReplay.nested_apply(
                select,
                self.actions,
            ),
            ExperienceReplay.nested_apply(
                select,
                self.rewards,
            ),
            ExperienceReplay.nested_apply(
                select,
                self.next_observations,
            ),
        )
