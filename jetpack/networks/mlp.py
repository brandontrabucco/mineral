"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod


class MLP(tf.keras.Model, ABC):

    def __init__(
        self,
        tau=1e-3,
        optimizer_class=tf.keras.optimizers.Adam,
        **optimizer_kwargs
    ):
        super(MLP, self).__init__()
        self.tau = tau
        self.optimizer = optimizer_class(**optimizer_kwargs)

    @abstractmethod
    def call(
        self,
        *inputs
    ):
        return NotImplemented

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        with tf.GradientTape() as gradient_tape:
            return gradient_tape.gradient(
                loss_function(*inputs),
                self.trainable_variables
            )

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

    def minimize(
        self,
        loss_function,
        *inputs
    ):
        self.apply_gradients(
            self.compute_gradients(
                loss_function,
                *inputs
            )
        )

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
