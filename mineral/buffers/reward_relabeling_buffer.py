"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.buffer import Buffer


class RewardRelabelingBuffer(Buffer):
    
    def __init__(
        self,
        buffer,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        order=2,
        **kwargs
    ):
        Buffer.__init__(self, **kwargs)
        self.buffer = buffer
        self.observation_selector = observation_selector
        self.goal_selector = goal_selector
        self.order = order

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
        (observations,
            actions,
            rewards,
            terminals) = self.buffer.sample(batch_size)
        error = (self.observation_selector(observations) -
                 self.goal_selector(observations))
        rewards = tf.linalg.norm(error, ord=self.order, axis=(-1))
        return (
            observations,
            actions,
            rewards[:, 1:],
            terminals)
