"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


def default_goal_assigner(goal, observation):
    observation["goal"] = goal
    return observation


class HindsightRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        time_skip=1,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        goal_assigner=default_goal_assigner,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.time_skip = time_skip
        self.observation_selector = observation_selector
        self.goal_selector = goal_selector
        self.goal_assigner = goal_assigner

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        selected_observations = self.observation_selector(
            observations)
        indices = (tf.tile(
            tf.expand_dims(tf.range(tf.shape(selected_observations)[1]), 0),
            [tf.shape(selected_observations)[0], 1])
                // self.time_skip) * self.time_skip

        achieved_goals = tf.gather(
            selected_observations,
            indices,
            batch_dims=1)
        original_goals = self.goal_selector(observations)

        relabel_condition = self.get_relabeled_mask(achieved_goals)
        relabeled_goals = tf.where(
            relabel_condition,
            achieved_goals,
            original_goals)

        relabeled_observations = self.goal_assigner(
            relabeled_goals, observations)
        return (
            relabeled_observations,
            actions,
            rewards,
            terminals)
