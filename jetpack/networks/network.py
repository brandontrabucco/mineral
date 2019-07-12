"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod
from jetpack.core.has_gradient import HasGradient
from jetpack.distributions.distribution import Distribution
from jetpack.distributions.gaussians.gaussian_distribution import GaussianDistribution


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
        self.distribution_class = distribution_class
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
                loss_function(), self.trainable_variables
            )

    def apply_gradients(
        self,
        gradients
    ):
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

    def soft_update(
        self,
        weights
    ):
        self.set_weights([
            self.tau * w + (1.0 - self.tau) * w_self
            for w, w_self in zip(weights, self.get_weights())
        ])

    def get_activations(self, *inputs):
        return self(*inputs)

    def get_parameters(self, *inputs):
        return self.distribution_class.get_parameters(self, *inputs)

    def sample(self, *inputs):
        return self.distribution_class.sample(self, *inputs)

    def get_expected_value(self, *inputs):
        return self.distribution_class.get_expected_value(self, *inputs)

    def get_log_probs(self, *inputs):
        return self.distribution_class.get_log_probs(self, *inputs)

    def get_kl_divergence(self, *inputs):
        return self.distribution_class.get_kl_divergence(self, *inputs)

    def get_fisher_information(self, *inputs):
        return self.distribution_class.get_fisher_information(self, *inputs)
