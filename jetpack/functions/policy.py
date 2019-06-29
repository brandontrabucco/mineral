"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def get_stochastic_actions(
        self,
        observations
    ):
        return NotImplemented

    @abstractmethod
    def get_deterministic_actions(
        self,
        observations
    ):
        return NotImplemented

    def get_probs(
        self,
        observations,
        actions
    ):
        return tf.exp(self.get_log_probs(
            observations,
            actions
        ))

    @abstractmethod
    def get_log_probs(
        self,
        observations,
        actions
    ):
        return NotImplemented

    @abstractmethod
    def get_kl_divergence(
        self,
        other_policy,
        observations
    ):
        return NotImplemented

    @abstractmethod
    def naturalize(
        self,
        observations,
        y,
    ):
        return NotImplemented
