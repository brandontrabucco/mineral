"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.relabelers.relabeler import Relabeler


def default_goal_assigner(goal, observation):
    observation["goal"] = goal
    return observation


class HindsightRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        achieved_goal_selector=(lambda x: x["achieved_goal"]),
        goal_selector=(lambda x: x["goal"]),
        goal_assigner=default_goal_assigner,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.achieved_goal_selector = achieved_goal_selector
        self.goal_selector = goal_selector
        self.goal_assigner = goal_assigner

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        achieved_goals = self.achieved_goal_selector(observations)
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
