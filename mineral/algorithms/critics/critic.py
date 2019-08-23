"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from abc import ABC, abstractmethod
from mineral.algorithms.base import Base


class Critic(Base, ABC):

    @abstractmethod
    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented

    @abstractmethod
    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented

    @abstractmethod
    def update_critic(
        self,
        observations,
        actions,
        rewards,
        terminals,
        bellman_target_values,
        discount_target_values
    ):
        return NotImplemented

    @abstractmethod
    def soft_update(
        self
    ):
        return NotImplemented

    def update_algorithm(
        self, 
        observations,
        actions,
        rewards,
        terminals
    ):
        bellman_target_values = self.bellman_target_values(
            observations,
            actions,
            rewards,
            terminals
        )
        discount_target_values = self.discount_target_values(
            observations,
            actions,
            rewards,
            terminals
        )
        self.update_critic(
            observations,
            actions,
            rewards,
            terminals,
            bellman_target_values,
            discount_target_values
        )
        self.soft_update()

    @abstractmethod
    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return NotImplemented
