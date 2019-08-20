"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.relabelers.relabeler import Relabeler


class ReachabilityRelabeler(Relabeler):
    
    def __init__(
        self,
        policy,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        reward_scale=1.0,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.policy = policy
        self.observation_selector = observation_selector
        self.reward_scale = reward_scale

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
            cumulative_distances += self.reward_scale * tf.linalg.norm(
                tf.reshape(error, [tf.shape(error)[1], tf.shape(error)[1], -1]),
                ord=self.order, axis=(-1))

        relabel_condition = self.relabel_probability >= tf.random.uniform(
            tf.shape(rewards),
            maxval=1.0,
            dtype=tf.float32)

        rewards = tf.where(
            relabel_condition, -cumulative_distances, rewards)
        return (
            observations,
            actions,
            rewards,
            terminals)
