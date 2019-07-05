"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from jetpack.algorithms.critics.critic import Critic


class TwinDelayedCritic(Critic):

    def __init__(
        self,
        critic1,
        critic2,
        **kwargs
    ):
        Critic.__init__(self, **kwargs)
        self.critic1 = critic1
        self.critic2 = critic2

    def bellman_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return tf.minimum(
            self.critic1.bellman_target_values(
                observations,
                actions,
                rewards,
                terminals
            ),
            self.critic2.bellman_target_values(
                observations,
                actions,
                rewards,
                terminals
            )
        )

    def discount_target_values(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return self.critic1.discount_target_values(
            observations,
            actions,
            rewards,
            terminals
        )

    def update_critic(
        self,
        observations,
        actions,
        rewards,
        terminals,
        bellman_target_values,
        discount_target_values
    ):
        self.critic1.update_critic(
            observations,
            actions,
            rewards,
            terminals,
            bellman_target_values,
            discount_target_values
        )
        self.critic2.update_critic(
            observations,
            actions,
            rewards,
            terminals,
            bellman_target_values,
            discount_target_values
        )

    def soft_update(
        self
    ):
        self.critic1.soft_update()
        self.critic2.soft_update()

    def get_advantages(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        return self.critic1.get_advantages(
            observations,
            actions,
            rewards,
            terminals
        )