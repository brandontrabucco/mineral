"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.samplers.sampler import Sampler


class HierarchicalSampler(Sampler):

    def __init__(
        self,
        *args,
        time_skips=(1,),
        **kwargs
    ):
        Sampler.__init__(self, *args, **kwargs)
        self.num_levels = len(self.policies)
        self.time_skips = time_skips + (
            1 for _i in range(self.num_levels - len(time_skips)))

    def push_through_hierarchy(
        self,
        hierarchy_samples,
        time_step,
        observation,
        random=False,
    ):
        for level in reversed(range(self.num_levels)):
            if time_step % np.prod(self.time_skips[:level]) == 0:
                policy_inputs = self.selector(level, observation)[np.newaxis, ...]
                if level < self.num_levels - 1:
                    policy_inputs = np.concatenate([
                        policy_inputs, hierarchy_samples[level + 1][2]], 0)
                if random:
                    current_action = self.policies[level].sample(
                        policy_inputs)[0, ...].numpy()
                else:
                    current_action = self.policies[level].get_expected_value(
                        policy_inputs)[0, ...].numpy()
                    hierarchy_samples[level][0] += 1
                hierarchy_samples[level][1] = {
                    **observation, "induced_actions": [], "induced_observations": []}
                if level < self.num_levels - 1:
                    hierarchy_samples[level][1]["goal"] = hierarchy_samples[level + 1][2]
                hierarchy_samples[level][2] = current_action
                hierarchy_samples[level][3] = 0.0
                if level < self.num_levels - 1:
                    hierarchy_samples[level + 1][1]["induced_actions"].append(current_action)
                    hierarchy_samples[level + 1][1]["induced_observations"].append(observation)

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
            hierarchy_samples = [[0, {}, None, 0.0] for _level in range(self.num_levels)]
            observation = self.env.reset()
            path_return = 0.0
            for time_step in range(self.buffers[0].max_path_length):
                self.push_through_hierarchy(hierarchy_samples, time_step, observation, random=random)
                next_observation, reward, done, info = self.env.step(hierarchy_samples[0][2])
                path_return = path_return + reward
                observation = next_observation
                for level in range(self.num_levels):
                    hierarchy_samples[level][3] += reward
                    if (save_paths and (
                            len(hierarchy_samples[level][1]["induced_actions"]) == self.time_skips[level] or
                            level == 0)):
                        self.buffers[level].insert_sample(*hierarchy_samples[level])
                if render:
                    self.env.render(**render_kwargs)
                if save_paths:
                    self.increment()
                if done:
                    break
            all_returns.append(path_return)
        return np.mean(all_returns) if len(all_returns) > 0 else 0
