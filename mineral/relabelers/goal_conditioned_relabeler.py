"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.relabelers import Relabeler


class GoalConditionedRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        order=2,
        reward_scale=1.0,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
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
        error = self.observation_selector(
            observations) - self.goal_selector(observations)
        goal_conditioned_rewards = -self.reward_scale * tf.linalg.norm(
            tf.reshape(error, [tf.shape(error)[0], tf.shape(error)[1], -1]),
            ord=self.order, axis=(-1))[:, 1:]

        rewards = tf.where(
            self.get_relabeled_mask(rewards), goal_conditioned_rewards, rewards)
        return (
            observations,
            actions,
            rewards,
            terminals)
