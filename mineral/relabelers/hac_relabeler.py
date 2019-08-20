"""Author: Brandon Trabucco, Copyright 2019"""


import tensorflow as tf
from mineral.relabelers import Relabeler


class HACRelabeler(Relabeler):
    
    def __init__(
        self,
        *args,
        observation_selector=(lambda x: x["proprio_observation"]),
        **kwargs
    ):
        Relabeler.__init__(self, *args, **kwargs)
        self.observation_selector = observation_selector

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

        relabeled_actions = induced_observations[-1][:, :(-1), ...]
        relabel_condition = self.get_relabeled_mask(actions)

        actions = tf.where(
            relabel_condition, relabeled_actions, actions)
        return (
            observations,
            actions,
            rewards,
            terminals)
