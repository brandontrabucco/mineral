"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


class SubgoalTestingRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        goal_selector=(lambda x: x["goal"]),
        order=2,
        threshold=0.1,
        penalty=(-1.0),
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.observation_selector = observation_selector
        self.goal_selector = goal_selector
        self.order = order
        self.threshold = threshold
        self.penalty = penalty

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        error = self.observation_selector(
            observations) - self.goal_selector(observations)
        goal_distances = self.reward_scale * tf.linalg.norm(
            tf.reshape(error, [tf.shape(error)[1], tf.shape(error)[1], -1]),
            ord=self.order, axis=(-1))[:, 1:]
        test_passed_condition = goal_distances < self.threshold
        tested_rewards = tf.where(
            test_passed_condition,
            rewards,
            rewards + self.penalty)
        relabel_condition = self.relabel_probability > tf.random.uniform(
            tf.shape(rewards),
            maxval=1.0,
            dtype=tf.float32)
        rewards = tf.where(
            relabel_condition, tested_rewards, rewards)
        return (
            observations,
            actions,
            rewards,
            terminals)
