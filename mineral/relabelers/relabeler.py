"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.core.buffers import Buffer
from abc import ABC, abstractmethod


class Relabeler(Buffer, ABC):
    
    def __init__(
        self,
        buffer,
        relabel_probability=1.0,
        **kwargs
    ):
        self.buffer = buffer
        self.relabel_probability = relabel_probability

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

    def request_head(
        self,
        *args,
        **kwargs
    ):
        self.buffer.request_head(*args, **kwargs)

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

    def get_relabeled_mask(
        self,
        data
    ):
        relabel_condition = tf.math.less_equal(
            tf.random.uniform(
                tf.shape(data)[:2],
                maxval=1.0,
                dtype=tf.float32), self.relabel_probability)
        while len(relabel_condition.shape) < len(data.shape):
            relabel_condition = tf.expand_dims(relabel_condition, -1)
        return tf.broadcast_to(
            relabel_condition, tf.shape(data))

    @abstractmethod
    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented
