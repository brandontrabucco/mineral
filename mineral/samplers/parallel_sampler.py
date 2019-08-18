"""Author: Brandon Trabucco, Copyright 2019"""


import threading
import numpy as np
from mineral.samplers.sampler import Sampler
from mineral.samplers.path_sampler import PathSampler


class ParallelSampler(Sampler):

    def __init__(
        self,
        *args,
        num_threads=4,
        max_path_length=256,
        num_warm_up_paths=1024,
        num_exploration_paths=32,
        num_evaluation_paths=32,
        **kwargs
    ):
        Sampler.__init__(
            self,
            max_path_length=max_path_length,
            num_warm_up_paths=num_warm_up_paths,
            num_exploration_paths=num_exploration_paths,
            num_evaluation_paths=num_evaluation_paths,
            **kwargs)
        self.num_threads = num_threads
        self.inner_samplers = [
            PathSampler(
                *args,
                max_path_length=max_path_length,
                num_warm_up_paths=(num_warm_up_paths % num_threads + num_warm_up_paths // num_threads),
                num_exploration_paths=(num_exploration_paths % num_threads + num_exploration_paths // num_threads),
                num_evaluation_paths=(num_evaluation_paths % num_threads + num_evaluation_paths // num_threads),
                **kwargs)]
        for _i in range(num_threads):
            self.inner_samplers.append(PathSampler(
                *args,
                max_path_length=max_path_length,
                num_warm_up_paths=num_warm_up_paths//num_threads,
                num_exploration_paths=num_exploration_paths//num_threads,
                num_evaluation_paths=num_evaluation_paths//num_threads,
                **kwargs))
        for inner_sampler in self.inner_samplers:
            inner_sampler.increment = self.increment

    def collect(
        self,
        thread_function
    ):
        reward_list = []
        threads = [threading.Thread(
            target=thread_function, args=(sampler, reward_list)) for sampler in self.inner_samplers]
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
            output_list.append(inner_sampler.warm_up(render=render, **render_kwargs))
        return self.collect(thread_function)

    def explore(
        self,
        render=False,
        **render_kwargs
    ):
        def thread_function(inner_sampler, output_list):
            output_list.append(inner_sampler.explore(render=render, **render_kwargs))
        return self.collect(thread_function)

    def evaluate(
        self,
        render=False,
        **render_kwargs
    ):
        def thread_function(inner_sampler, output_list):
            output_list.append(inner_sampler.evaluate(render=render, **render_kwargs))
        return self.collect(thread_function)
