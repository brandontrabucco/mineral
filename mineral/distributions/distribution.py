"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod


class Distribution(ABC):

    def get_probs(
        self,
        *inputs,
        **kwargs
    ):
        return tf.exp(self.get_log_probs(
            *inputs
        ))

    @abstractmethod
    def get_activations(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_parameters(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def sample_from_prior(
        self,
        shape,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_expected_value(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_expected_value_of_prior(
        self,
        shape,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_log_probs(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_kl_divergence(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented

    @abstractmethod
    def get_fisher_information(
        self,
        *inputs,
        **kwargs
    ):
        return NotImplemented
