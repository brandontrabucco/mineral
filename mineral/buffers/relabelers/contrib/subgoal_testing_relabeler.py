"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


class SubgoalTestingRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        order=2,
        threshold=0.1,
        penalty=(-1.0),
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.observation_selector = observation_selector
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
        induced_observations = [
            self.observation_selector(x)
            for x in observations["induced_observations"]]

        cumulative_distances = 0.0
        for lower_observation in induced_observations:
            error = lower_observation[:, 1:, ...] - actions
            cumulative_distances += tf.linalg.norm(
                tf.reshape(error, [tf.shape(error)[0], tf.shape(error)[1], -1]),
                ord=self.order, axis=(-1))

        test_passed_condition = cumulative_distances < self.threshold
        tested_rewards = tf.where(
            test_passed_condition,
            rewards,
            rewards + self.penalty)

        rewards = tf.where(
            self.get_relabeled_mask(rewards), tested_rewards, rewards)
        return (
            observations,
            actions,
            rewards,
            terminals)
