"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import mineral as jp
from mineral.buffers.buffer import Buffer


class PathBuffer(Buffer):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        Buffer.__init__(self, *args, **kwargs)
        self.size = 0
        self.head = 0
        self.tail = np.zeros([self.max_size], dtype=np.int32)
        self.observations = None
        self.actions = None
        self.rewards = None

    def reset(
        self
    ):
        self.size = 0
        self.head = 0
        self.tail = np.zeros([self.max_size], dtype=np.int32)

    def inflate(
        self,
        observation,
        action,
        reward
    ):
        def inflate_backend(x):
            return np.zeros([self.max_size, self.max_path_length,
                             *x.shape], dtype=np.float32)
        self.observations = jp.nested_apply(inflate_backend, observation)
        self.actions = jp.nested_apply(inflate_backend, action)
        self.rewards = jp.nested_apply(inflate_backend, reward)

    def insert_sample(
        self,
        j,
        observation,
        action,
        reward
    ):
        def insert_sample_backend(x, y):
            x[self.head, j, ...] = y
        if self.size == 0:
            self.inflate(observation, action, reward)
        jp.nested_apply(insert_sample_backend, self.observations, observation)
        jp.nested_apply(insert_sample_backend, self.actions, action)
        jp.nested_apply(insert_sample_backend, self.rewards, reward)
        self.tail[self.head] = j + 1

    def finish_path(self):
        self.head = (self.head + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size
    ):
        indices = np.random.choice(
            self.size, size=batch_size, replace=(self.size < batch_size))
        observations = jp.nested_apply(lambda x: x[indices, ...], self.observations)
        actions = jp.nested_apply(
            lambda x: x[indices, :(-1), ...], self.actions)
        rewards = jp.nested_apply(
            lambda x: x[indices, :(-1), ...], self.rewards)
        lengths = jp.nested_apply(
            lambda x: x[indices, ...], self.tail)
        max_lengths = np.arange(self.max_path_length)[np.newaxis, :]
        terminals = (lengths[:, np.newaxis] - 1 >= max_lengths).astype(np.float32)
        rewards = terminals[:, :(-1)] * rewards
        return (self.selector(observations),
                actions,
                rewards,
                terminals)
