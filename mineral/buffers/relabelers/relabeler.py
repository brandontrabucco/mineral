"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.buffers.buffer import Buffer
from abc import ABC, abstractmethod


class Relabeler(Buffer, ABC):
    
    def __init__(
        self,
        buffer
    ):
        self.buffer = buffer

    def __getattr__(
        self,
        attr
    ):
        return getattr(self.buffer, attr)

    def inflate(
        self,
        *args,
        **kwargs
    ):
        self.buffer.inflate(*args, **kwargs)

    def insert_sample(
        self,
        *args,
        **kwargs
    ):
        self.buffer.insert_sample(*args, **kwargs)

    def finish_path(
        self,
        *args,
        **kwargs
    ):
        self.buffer.finish_path(*args, **kwargs)

    def reset(
        self,
        *args,
        **kwargs
    ):
        self.buffer.reset(*args, **kwargs)

    def sample(
        self,
        batch_size
    ):
        return self.relabel(*self.buffer.sample(batch_size))

    @abstractmethod
    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented
