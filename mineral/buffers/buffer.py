"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Buffer(ABC):
    
    def __init__(
        self, 
        env,
        policy,
        max_size=1000,
        max_path_length=256,
        selector=None,
        monitor=None
    ):
        self.env = env
        self.policy = policy
        self.selector = (lambda x: x) if selector is None else selector
        self.monitor = monitor
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.num_steps_collected = 0

    def increment(self):
        self.num_steps_collected += 1
        if self.monitor is not None:
            self.monitor.set_step(self.num_steps_collected)

    @abstractmethod
    def reset(
        self,
    ):
        return NotImplemented

    @abstractmethod
    def collect(
        self,
        num_paths_to_collect=1,
        save_paths=True,
        render=False,
        **render_kwargs
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        batch_size
    ):
        return NotImplemented
