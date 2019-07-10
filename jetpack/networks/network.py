"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod
from jetpack.has_gradient import HasGradient
from jetpack.distributions.distribution import Distribution
from jetpack.distributions.gaussian_distribution import GaussianDistribution


class Network(tf.keras.Model, Distribution, HasGradient, ABC):

    def __init__(
        self,
        tau=1e-3,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={},
        distribution_class=GaussianDistribution,
        distribution_kwargs={}
    ):
        tf.keras.Model.__init__(self)
        distribution_class.__init__(self, **distribution_kwargs)
        self.tau = tau
        self.optimizer = optimizer_class(**optimizer_kwargs)

    @abstractmethod
    def call(
        self,
        *inputs
    ):
        return NotImplemented

    def get_activations(
        self,
        *inputs
    ):
        return self(*inputs)

    def compute_gradients(
        self,
        loss_function,
        *inputs
    ):
        with tf.GradientTape() as gradient_tape:
            return gradient_tape.gradient(
                loss_function(),
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
