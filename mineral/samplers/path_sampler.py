"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.samplers.sampler import Sampler


class PathSampler(Sampler):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        Sampler.__init__(self, *args, **kwargs)

    def collect(
        self,
        num_samples_to_collect,
        random=False,
        save_paths=False,
        render=False,
        **render_kwargs
    ):
        all_returns = []
        for i in range(min(num_samples_to_collect, self.buffer.max_size)):
            observation = self.env.reset()
            path_return = 0.0
            for j in range(self.buffer.max_path_length):
                if random:
                    action = self.policy.sample(
                        self.selector(observation)[np.newaxis, ...])[0, ...].numpy()
                else:
                    action = self.policy.get_expected_value(
                        self.selector(observation)[np.newaxis, ...])[0, ...].numpy()
                next_observation, reward, done, info = self.env.step(action)
                if render:
                    self.env.render(**render_kwargs)
                if save_paths:
                    self.increment()
                    self.buffer.insert_sample(j, observation, action, reward)
                path_return = path_return + reward
                if done:
                    break
                observation = next_observation
            if save_paths:
                self.buffer.finish_path()
            all_returns.append(path_return)
        return (np.mean(all_returns)
                if len(all_returns) > 0 else 0)
