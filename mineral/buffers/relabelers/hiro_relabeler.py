"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.buffers.relabelers.relabeler import Relabeler


class HIRORelabeler(Relabeler):
    
    def __init__(
        self,
        lower_level_policy,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        num_samples=8,
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.lower_level_policy = lower_level_policy
        self.observation_selector = observation_selector
        self.num_samples = num_samples

    def relabel(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        induced_actions = observations["induced_actions"]
        induced_observations = [
            self.observation_selector(x)
            for x in observations["induced_observations"]]
        achieved_goal = induced_observations[-1][:, :(-1), ...]
        candidates = achieved_goal + tf.random.normal(
            (self.num_samples,) + achieved_goal.shape)
        candidates = tf.concat([
            tf.expand_dims(actions, 0),
            tf.expand_dims(achieved_goal, 0),
            candidates], 0)

        log_probabilities = 0.0
        for lower_actions, lower_observations in zip(
                induced_actions, induced_observations):
            lower_actions = tf.tile(
                tf.expand_dims(lower_actions[:, :(-1), ...], 0),
                [self.num_samples + 2] + [
                    1 for _x in tf.shape(lower_actions)])
            lower_observations = tf.tile(
                tf.expand_dims(lower_observations[:, :(-1), ...], 0),
                [self.num_samples + 2] + [
                    1 for _x in tf.shape(lower_observations)])

            log_probabilities = log_probabilities + (
                self.lower_level_policy.get_log_probs(
                    lower_actions,
                    tf.concat([lower_observations, candidates], -1)))

        indices = tf.argmax(
            log_probabilities, axis=0, output_type=tf.int32)
        relabeled_actions = tf.squeeze(
            tf.gather(
                tf.transpose(candidates, [1, 2, 0, 3]),
                tf.expand_dims(indices, 2),
                batch_dims=2), 2)

        relabel_condition = tf.broadcast_to(
            self.relabel_probability > tf.random.uniform(
                tf.shape(actions)[:2],
                maxval=1.0,
                dtype=tf.float32),
            tf.shape(actions))
        actions = tf.where(
            relabel_condition, relabeled_actions, actions)
        return (
            observations,
            actions,
            rewards,
            terminals)
