"""Author: Brandon Trabucco, Copyright 2019"""


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

    @abstractmethod
    def get_probs(
        self,
        observations,
        actions
    ):
        return NotImplemented

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
    def fisher_vector_product(
        self,
        observations,
        y
    ):
        return NotImplemented

    @abstractmethod
    def inverse_fisher_vector_product(
        self,
        observations,
        g
    ):
        return NotImplemented
