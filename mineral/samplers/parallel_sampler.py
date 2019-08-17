"""Author: Brandon Trabucco, Copyright 2019"""


import threading
import numpy as np
from mineral.samplers.sampler import Sampler
from mineral.samplers.path_sampler import PathSampler


class ParallelSampler(Sampler):

    def __init__(
        self,
        *args,
        max_path_length=256,
        num_warm_up_paths=1024,
        num_exploration_paths=32,
        num_evaluation_paths=32,
        num_threads=4,
        **kwargs
    ):
        Sampler.__init__(
            self,
            *args,
            max_path_length=max_path_length,
            num_warm_up_paths=num_warm_up_paths,
            num_exploration_paths=num_exploration_paths,
            num_evaluation_paths=num_evaluation_paths,
            **kwargs)
        self.num_threads = num_threads
        self.inner_samplers = [PathSampler(
            *args,
            max_path_length=(max_path_length % self.num_threads + max_path_length // self.num_threads),
            num_warm_up_paths=(num_warm_up_paths % self.num_threads + num_warm_up_paths // self.num_threads),
            num_exploration_paths=(num_exploration_paths % self.num_threads + num_exploration_paths // self.num_threads),
            num_evaluation_paths=(num_evaluation_paths % self.num_threads + num_evaluation_paths // self.num_threads),
            **kwargs)]
        self.inner_samplers += [PathSampler(
            *args,
            max_path_length=max_path_length//self.num_threads,
            num_warm_up_paths=num_warm_up_paths//self.num_threads,
            num_exploration_paths=num_exploration_paths//self.num_threads,
            num_evaluation_paths=num_evaluation_paths//self.num_threads,
            **kwargs) for _i in range(1, num_threads)]
        for inner_sampler in self.inner_samplers:
            inner_sampler.increment = self.increment

    def collect(
        self,
        num_samples_to_collect,
        random=False,
        save_paths=False,
        render=False,
        **render_kwargs
    ):
        pass

    def parallel_collect(
        self,
        thread_function
    ):
        reward_list = []
        threads = [threading.Thread(
            target=thread_function,
            args=(sampler, reward_list)) for sampler in self.inner_samplers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return np.mean(reward_list)

    def warm_up(
        self,
        render=False,
        **render_kwargs
    ):
        def thread_function(inner_sampler, output_list):
            output_list.append(inner_sampler.warp_up(render=render, **render_kwargs))
        return self.parallel_collect(thread_function)

    def explore(
        self,
        render=False,
        **render_kwargs
    ):
        def thread_function(inner_sampler, output_list):
            output_list.append(inner_sampler.explore(render=render, **render_kwargs))
        return self.parallel_collect(thread_function)

    def evaluate(
        self,
        render=False,
        **render_kwargs
    ):
        def thread_function(inner_sampler, output_list):
            output_list.append(inner_sampler.evaluate(render=render, **render_kwargs))
        return self.parallel_collect(thread_function)
