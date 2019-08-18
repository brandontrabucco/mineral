"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod
from mineral.core.has_gradient import HasGradient
from mineral.core.cloneable import Cloneable
from mineral.distributions.distribution import Distribution
from mineral.distributions.gaussians.gaussian import Gaussian


class Network(tf.keras.Model, Distribution, HasGradient, Cloneable, ABC):

    def __init__(
        self,
        tau=1e-3,
        optimizer_class=tf.keras.optimizers.Adam,
        optimizer_kwargs={},
        distribution_class=Gaussian,
        distribution_kwargs={},
        **kwargs
    ):
        tf.keras.Model.__init__(self)
        distribution_class.__init__(self, **distribution_kwargs)
        Cloneable.__init__(
            self,
            tau=tau,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs)
        self.tau = tau
        self.distribution_class = distribution_class
        self.optimizer = optimizer_class(**optimizer_kwargs)

    @abstractmethod
    def call(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    def copy_to(
        self,
        clone
    ):
        weights = self.get_weights()
        clone_weights = clone.get_weights()
        if len(weights) > 0 and len(weights) == len(clone_weights):
            clone.set_weights(weights)

    def compute_gradients(
        self,
        loss_function,
        *inputs,
        **kwargs
    ):
        with tf.GradientTape() as gradient_tape:
            return gradient_tape.gradient(loss_function(), self.trainable_variables)

    def apply_gradients(
        self,
        gradients
    ):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def soft_update(
        self,
        weights
    ):
        self.set_weights([self.tau * w + (1.0 - self.tau) * w_self
                          for w, w_self in zip(weights, self.get_weights())])

    def get_activations(self, *inputs, **kwargs):
        return self(*inputs, **kwargs)

    def get_parameters(self, *inputs, **kwargs):
        return self.distribution_class.get_parameters(self, *inputs, **kwargs)

    def sample(self, *inputs, **kwargs):
        return self.distribution_class.sample(self, *inputs, **kwargs)

    def sample_from_prior(self, shape, **kwargs):
        return self.distribution_class.sample_from_prior(self, shape, **kwargs)

    def get_expected_value(self, *inputs, **kwargs):
        return self.distribution_class.get_expected_value(self, *inputs, **kwargs)

    def get_expected_value_from_prior(self, shape, **kwargs):
        return self.distribution_class.get_expected_value_from_prior(self, shape, **kwargs)

    def get_log_probs(self, *inputs, **kwargs):
        return self.distribution_class.get_log_probs(self, *inputs, **kwargs)

    def get_kl_divergence(self, *inputs, **kwargs):
        return self.distribution_class.get_kl_divergence(self, *inputs, **kwargs)

    def get_fisher_information(self, *inputs, **kwargs):
        return self.distribution_class.get_fisher_information(self, *inputs, **kwargs)
