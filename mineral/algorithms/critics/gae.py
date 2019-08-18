"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.algorithms.critics.critic import Critic
from mineral import discounted_sum


class GAE(Critic):

    def __init__(
        self,
        critic,
        gamma=0.99,
        lamb=0.95,
        **kwargs
    ):
        Critic.__init__(self, **kwargs)
        self.critic = critic
        self.gamma = gamma
        self.lamb = lamb

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return self.critic.bellman_target_values(
            observations,
            actions,
            rewards,
            terminals)

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return self.critic.discount_target_values(
            observations,
            actions,
            rewards,
            terminals)

    def update_critic(
        self,
        observations,
        actions,
        rewards,
        terminals,
        bellman_target_values,
        discount_target_values
    ):
        self.critic.update_critic(
            observations,
            actions,
            rewards,
            terminals,
            bellman_target_values,
            discount_target_values)

    def soft_update(
        self
    ):
        self.critic.soft_update()

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        advantages = self.critic.get_advantages(
            observations,
            actions,
            rewards,
            terminals)
        advantages = discounted_sum(advantages, self.gamma * self.lamb)
        self.record("generalized_advantages_mean", tf.reduce_mean(advantages))
        return advantages
