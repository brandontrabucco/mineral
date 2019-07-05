"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Buffer(ABC):
    
    def __init__(
        self, 
        env,
        policy
    ):
        self.env = env
        self.policy = policy

    @abstractmethod
    def reset(
        self,
        max_size,
        max_path_length
    ):
        return NotImplemented

    @abstractmethod
    def collect(
        self,
        num_paths_to_collect,
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
