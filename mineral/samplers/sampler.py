"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(
        self,
        env,
        *inputs,
        num_warm_up_samples=1024,
        num_exploration_samples=32,
        num_evaluation_samples=32,
        selector=None,
        monitor=None
    ):
        self.env = env
        self.policies = inputs[0::2]
        self.buffers = inputs[1::2]
        self.num_warm_up_samples = num_warm_up_samples
        self.num_exploration_samples = num_exploration_samples
        self.num_evaluation_samples = num_evaluation_samples
        self.selector = (lambda i, x: x) if selector is None else selector
        self.monitor = monitor
        self.num_steps_collected = 0

    def increment(self):
        self.num_steps_collected += 1
        if self.monitor is not None:
            self.monitor.set_step(self.num_steps_collected)

    def warm_up(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(
            self.num_warm_up_samples,
            random=True,
            save_paths=True,
            render=render,
            **render_kwargs)

    def explore(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(
            self.num_exploration_samples,
            random=True,
            save_paths=True,
            render=render,
            **render_kwargs)

    def evaluate(
        self,
        render=False,
        **render_kwargs
    ):
        return self.collect(
            self.num_evaluation_samples,
            random=False,
            save_paths=False,
            render=render,
            **render_kwargs)

    def reset(
        self,
    ):
        return [b.reset() for b in self.buffers]

    @abstractmethod
    def collect(
        self,
        num_samples_to_collect,
        random=False,
        save_paths=False,
        render=False,
        **render_kwargs
    ):
        return NotImplemented
