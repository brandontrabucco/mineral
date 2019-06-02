"""Author: Brandon Trabucco, Copyright 2019"""


from abc import ABC, abstractmethod
import tensorflow as tf


class MLP(tf.keras.models.Model, ABC):

    def __init__(
        self,
        tau=0.9,
        optimizer=tf.keras.optimizers.Adam()
    ):
        super(MLP, self).__init__()
        self.tau = tau
        self.optimizer = optimizer

    def soft_update(
        self, 
        weights
    ):
        self.set_weights([
            self.tau * w + (1.0 - self.tau) * w_self
            for w, w_self in zip(
                weights,
                self.get_weights()
            )
        ])

    def apply_gradients(
        self, 
        gradients
    ):
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_variables
            )
        )

    @abstractmethod
    def __call__(
        self, 
        *inputs
    ):
        return NotImplemented