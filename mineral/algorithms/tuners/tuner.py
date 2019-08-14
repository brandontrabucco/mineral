"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC
from mineral.algorithms.base import Base


class Tuner(Base, ABC):

    def __init__(
        self,
        target=-1.0,
        initial_value=1.0,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={},
        **kwargs
    ):
        Base.__init__(self, **kwargs)
        self.target = target
        self.optimizer = optimizer_class(**optimizer_kwargs)
        self.tuning_variable = tf.Variable(initial_value)

    def get_tuning_variable(self):
        return self.tuning_variable
