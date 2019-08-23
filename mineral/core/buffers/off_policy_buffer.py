"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as ml
from mineral.core.buffers.buffer import Buffer


class OffPolicyBuffer(Buffer):

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
        def join(first, second):
            return np.concatenate([first, second], 1)
        paths = np.arange(self.max_size)
        path_indices = np.arange(self.max_path_length)
        candidates = np.stack(np.meshgrid(path_indices, paths), axis=(-1)).reshape(-1, 2)
        terminals = (self.tail[:, np.newaxis] - 1 > path_indices).astype(np.float32)
        indices = np.random.choice(
            candidates.shape[0],
            size=batch_size,
            p=terminals.reshape(-1) / terminals.sum(),
            replace=(candidates.shape[0] < batch_size))
        indices = np.take(candidates, indices, axis=0)
        first_observations = ml.nested_apply(
            lambda x: x[indices[:, 1], np.newaxis, indices[:, 0], ...],
            self.observations)
        next_observations = ml.nested_apply(
            lambda x: x[indices[:, 1], np.newaxis, indices[:, 0] + 1, ...],
            self.observations)
        observations = ml.nested_apply(join, first_observations, next_observations)
        actions = self.actions[indices[:, 1], np.newaxis, indices[:, 0], ...]
        rewards = self.rewards[indices[:, 1], np.newaxis, indices[:, 0], ...]
        terminals = np.ones([rewards.shape[0], 2])
        return self.selector(observations), actions, rewards, terminals
