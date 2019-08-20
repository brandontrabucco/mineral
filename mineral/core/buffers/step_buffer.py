"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as ml
from mineral.core.buffers.buffer import Buffer


class StepBuffer(Buffer):

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

    def reset(
        self
    ):
        self.buffer.reset()

    def inflate(
        self,
        observation,
        action,
        reward
    ):
        self.buffer.inflate(
            observation,
            action,
            reward)

    def insert_sample(
        self,
        head,
        tail,
        observation,
        action,
        reward
    ):
        self.buffer.insert_sample(
            head,
            tail,
            observation,
            action,
            reward)

    def request_head(
        self
    ):
        return self.buffer.request_head()

    def sample(
        self,
        batch_size
    ):
        paths = np.arange(self.max_size)
        path_indices = np.arange(self.max_path_length)
        candidates = np.stack(*np.meshgrid(paths, path_indices), axis=(-1)).reshape(-1, 2)
        lengths = ml.nested_apply(lambda x: x[indices, ...], self.tail)
        terminals = (lengths[:, np.newaxis] - 1 > path_indices).astype(np.float32).reshape(-1)
        indices = np.random.choice(
            self.max_size * self.max_path_length,
            size=batch_size,
            p=terminals,
            replace=(self.max_size * self.max_path_length < batch_size))
        indices = np.take(candidates, indices, axis=0)
        observations = ml.nested_apply(
            lambda x: x[indices[:, 0], indices[:, 1]:(indices[:, 1] + 2), ...],
            self.observations)
        actions = ml.nested_apply(
            lambda x: x[indices[:, 0], indices[:, 1]:(indices[:, 1] + 1), ...],
            self.actions)
        rewards = ml.nested_apply(
            lambda x: x[indices[:, 0], indices[:, 1]:(indices[:, 1] + 1), ...],
            self.rewards)
        terminals = terminals[indices[:, 0], indices[:, 1]:(indices[:, 1] + 2)]
        rewards = terminals[:, :(-1)] * rewards
        return self.selector(observations), actions, rewards, terminals
