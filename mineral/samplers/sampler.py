"""Author: Brandon Trabucco, Copyright 2019"""


import time
import copy
from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(
        self,
        max_path_length=256,
        num_warm_up_paths=1024,
        num_exploration_paths=32,
        num_evaluation_paths=32,
        selector=None,
        monitor=None,
        **kwargs
    ):
        self.max_path_length = max_path_length
        self.num_warm_up_paths = num_warm_up_paths
        self.num_exploration_paths = num_exploration_paths
        self.num_evaluation_paths = num_evaluation_paths
        self.selector = (lambda i, x: x) if selector is None else selector
        self.monitor = monitor
        self.num_steps_collected = 0
        self.begin_time = time.time()

    def increment(
        self
    ):
        self.num_steps_collected += 1
        elapsed = time.time() - self.begin_time
        if self.monitor is not None:
            self.monitor.set_step(self.num_steps_collected)
            self.monitor.record("steps_time", elapsed)
            self.monitor.record(
                "steps_per_second", self.num_steps_collected / elapsed)

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
        return self.collect(self.num_evaluation_paths, random=True,
                            save_paths=True, render=render, **render_kwargs)

    @abstractmethod
    def collect(
        self,
        *args,
        **kwargs
    ):
        return NotImplemented
