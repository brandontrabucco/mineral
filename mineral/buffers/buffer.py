"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod


class Buffer(ABC):
    
    def __init__(
        self,
        max_size=1024,
        max_path_length=10,
        selector=None,
        monitor=None
    ):
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.selector = (lambda x: x) if selector is None else selector
        self.monitor = monitor

    @abstractmethod
    def inflate(
        self,
        observation,
        action,
        reward
    ):
        return NotImplemented

    @abstractmethod
    def insert_sample(
        self,
        j,
        observation,
        action,
        reward
    ):
        return NotImplemented

    @abstractmethod
    def finish_path(
        self,
    ):
        return NotImplemented

    @abstractmethod
    def reset(
        self,
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        batch_size
    ):
        return NotImplemented
