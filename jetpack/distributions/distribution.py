"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod


class Distribution(ABC):

    def get_probs(
        self,
        *inputs
    ):
        return tf.exp(self.get_log_probs(
            *inputs
        ))

    @abstractmethod
    def get_activations(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def get_parameters(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def get_expected_value(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def get_log_probs(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def get_kl_divergence(
        self,
        *inputs
    ):
        return NotImplemented

    @abstractmethod
    def get_fisher_information(
        self,
        *inputs
    ):
        return NotImplemented
