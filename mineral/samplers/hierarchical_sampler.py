"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.samplers.sampler import Sampler


class HierarchicalSampler(Sampler):

    def __init__(
        self,
        time_skips,
        *args,
        selector=(lambda level, x: x),
        **kwargs
    ):
        Sampler.__init__(self, *args, selector=selector, **kwargs)
        self.time_skips = time_skips
        self.num_levels = len(time_skips)

    def reset(
        self,
    ):
        return [self.buffer[level].reset() for level in range(self.num_levels)]

    def collect(
        self,
        num_samples_to_collect,
        random=False,
        save_paths=False,
        render=False,
        **render_kwargs
    ):
        all_returns = []
        for i in range(num_samples_to_collect):
            intermediate_samples = [[
                0, {}, None, 0.0] for _level in range(self.num_levels)]
            observation = self.env.reset()
            path_return = 0.0
            for time_step in range(self.buffer.max_path_length):
                if render:
                    self.env.render(**render_kwargs)
                if save_paths:
                    self.increment()
                for level in reversed(range(self.num_levels)):
                    if time_step % np.prod(self.time_skips[:level]) == 0:
                        policy_inputs = self.selector(level, observation)[np.newaxis, ...]
                        if level < self.num_levels - 1:
                            policy_inputs = np.concatenate([
                                policy_inputs, intermediate_samples[level + 1][2]], 0)
                        if random:
                            current_action = self.policy[level].sample(
                                policy_inputs)[0, ...].numpy()
                        else:
                            current_action = self.policy[level].get_expected_value(
                                policy_inputs)[0, ...].numpy()
                        if save_paths and intermediate_samples[level][2] is not None:
                            self.buffer[level].insert_sample(*intermediate_samples[level])
                            intermediate_samples[level][0] += 1
                        intermediate_samples[level][1] = {
                            **observation,
                            "induced_actions": [],
                            "induced_observations": []}
                        if level < self.num_levels - 1:
                            intermediate_samples[level][1][
                                "goal"] = intermediate_samples[level + 1][2]
                        intermediate_samples[level][2] = current_action
                        intermediate_samples[level][3] = 0.0
                        if level < self.num_levels - 1:
                            intermediate_samples[level + 1][1][
                                "induced_actions"].append(current_action)
                            intermediate_samples[level + 1][1][
                                "induced_observations"].append(observation)
                next_observation, reward, done, info = self.env.step(
                    intermediate_samples[0][2])
                for level in range(self.num_levels):
                    intermediate_samples[level][3] += reward
                path_return = path_return + reward
                if done:
                    break
                observation = next_observation
            for level in range(self.num_levels):
                if save_paths and intermediate_samples[level][2] is not None:
                    self.buffer[level].insert_sample(*intermediate_samples[level])
                    self.buffer[level].finish_path()
            all_returns.append(path_return)
        return (np.mean(all_returns)
                if len(all_returns) > 0 else 0)
