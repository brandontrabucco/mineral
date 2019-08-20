"""Author: Brandon Trabucco, Copyright 2019"""


import time
from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(
        self,
        max_path_length=100,
        num_warm_up_paths=20,
        num_exploration_paths=20,
        num_evaluation_paths=20,
        selector=None,
        monitor=None,
        logging_interval=100,
        **kwargs
    ):
        self.max_path_length = max_path_length
        self.num_warm_up_paths = num_warm_up_paths
        self.num_exploration_paths = num_exploration_paths
        self.num_evaluation_paths = num_evaluation_paths
        self.selector = (lambda i, x: x) if selector is None else selector
        self.monitor = monitor
        self.logging_interval = logging_interval
        self.num_steps_collected = 0
        self.begin_time = time.time()

    def increment(
        self
    ):
        self.num_steps_collected += 1
        if self.monitor is not None:
            self.monitor.set_step(self.num_steps_collected)
            if self.num_steps_collected % self.logging_interval == 0:
                elapsed = time.time() - self.begin_time
                self.monitor.record("sampler_steps_time", elapsed)
                self.monitor.record(
                    "sampler_steps_per_second", self.num_steps_collected / elapsed)

    def warm_up(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(self.num_warm_up_paths, random=True,
                            save_paths=True, render=render, **render_kwargs)

    def explore(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(self.num_exploration_paths, random=True,
                            save_paths=True, render=render, **render_kwargs)

    def evaluate(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(self.num_evaluation_paths, random=False,
                            save_paths=False, render=render, **render_kwargs)

    @abstractmethod
    def collect(
        self,
        *args,
        **kwargs
    ):
        return NotImplemented
