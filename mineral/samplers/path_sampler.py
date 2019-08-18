"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.samplers.sampler import Sampler


class PathSampler(Sampler):

    def __init__(
        self,
        env,
        policies,
        buffers,
        time_skips=(1,),
        **kwargs
    ):
        Sampler.__init__(self, **kwargs)
        self.env = env.clone()
        self.master_policies = policies if isinstance(policies, list) else [policies]
        self.worker_policies = [p.clone() for p in self.master_policies]
        self.buffers = buffers if isinstance(buffers, list) else [buffers]
        self.num_levels = len(self.worker_policies)
        self.time_skips = tuple(time_skips) + tuple(
            1 for _i in range(self.num_levels - len(time_skips)))
        self.push_through_hierarchy([[-1, {}, None, 0.0] for _level in range(self.num_levels)],
                                    0, self.env.reset(), random=True)

    def push_through_hierarchy(
        self,
        hierarchy_samples,
        time_step,
        observation,
        random=False,
    ):
        for level in reversed(range(self.num_levels)):
            observation_for_this_level = {**observation}
            if time_step % np.prod(self.time_skips[:level + 1]) == 0:
                if level < self.num_levels - 1:
                    observation_for_this_level["goal"] = hierarchy_samples[level + 1][2]
                policy_inputs = self.selector(
                    level, observation_for_this_level)[np.newaxis, ...]
                if random:
                    current_action = self.worker_policies[level].sample(
                        policy_inputs)[0, ...].numpy()
                else:
                    current_action = self.worker_policies[level].get_expected_value(
                        policy_inputs)[0, ...].numpy()
                hierarchy_samples[level][0] += 1
                hierarchy_samples[level][1] = observation_for_this_level
                hierarchy_samples[level][2] = current_action
                hierarchy_samples[level][3] = 0.0
                if level > 0:
                    hierarchy_samples[level][1]["induced_actions"] = []
                    hierarchy_samples[level][1]["induced_observations"] = []
                if level < self.num_levels - 1:
                    hierarchy_samples[level + 1][1]["induced_actions"].append(current_action)
                    hierarchy_samples[level + 1][1]["induced_observations"].append(observation_for_this_level)

    def collect(
        self,
        num_samples_to_collect,
        random=False,
        save_paths=False,
        render=False,
        **render_kwargs
    ):
        all_rewards = []
        for master_p, worker_p in zip(self.master_policies, self.worker_policies):
            master_p.copy_to(worker_p)
        for i in range(num_samples_to_collect):
            hierarchy_samples = [[-1, {}, None, 0.0] for _level in range(self.num_levels)]
            heads = [self.buffers[level].request_head() for level in range(self.num_levels)]
            observation = self.env.reset()
            for time_step in range(self.max_path_length):
                self.push_through_hierarchy(hierarchy_samples, time_step, observation, random=random)
                next_observation, reward, done, info = self.env.step(hierarchy_samples[0][2])
                observation = next_observation
                all_rewards.append(reward)
                for level in range(self.num_levels):
                    hierarchy_samples[level][3] += reward
                    sample = hierarchy_samples[level][1]
                    if (save_paths and (
                            len(sample["induced_actions"]) == self.time_skips[level] or
                            "induced_actions" not in sample)):
                        self.buffers[level].insert_sample(heads[level], *hierarchy_samples[level])
                if render:
                    self.env.render(**render_kwargs)
                if save_paths:
                    self.increment()
                if done:
                    break
        return np.mean(all_rewards) if len(all_rewards) > 0 else 0.0
