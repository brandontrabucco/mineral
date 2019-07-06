"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def get_stochastic_observations(
        self,
        observations,
        actions
    ):
        return NotImplemented

    @abstractmethod
    def get_deterministic_observations(
        self,
        observations,
        actions
    ):
        return NotImplemented

    def get_probs(
        self,
        observations,
        actions,
        next_observations
    ):
        return tf.exp(self.get_log_probs(
            observations,
            actions,
            next_observations
        ))

    @abstractmethod
    def get_log_probs(
        self,
        observations,
        actions,
        next_observations
    ):
        return NotImplemented

    @abstractmethod
    def get_kl_divergence(
        self,
        other_model,
        observations,
        actions
    ):
        return NotImplemented
