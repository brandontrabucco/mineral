"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


class RewardRelabeler(Relabeler):
    
    def __init__(
        self,
        buffer,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        order=2,
        reward_scale=1.0
    ):
        Relabeler.__init__(self, buffer)
        self.observation_selector = observation_selector
        self.goal_selector = goal_selector
        self.order = order
        self.reward_scale = reward_scale

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        error = self.observation_selector(observations) - self.goal_selector(observations)
        rewards = -self.reward_scale * tf.linalg.norm(error, ord=self.order, axis=(-1))
        return (
            observations,
            actions,
            rewards[:, 1:],
            terminals)
