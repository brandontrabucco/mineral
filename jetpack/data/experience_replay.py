"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from jetpack.networks.policy import Policy


class ExperienceReplay(ABC):

    @staticmethod
    def nested_apply(
        function,
        *structures,
    ):
        if (isinstance(structures[0], np.ndarray) or 
                isinstance(structures[0], tf.Tensor) or not 
                isinstance(structures[0], list) or not 
                isinstance(structures[0], tuple) or not 
                isinstance(structures[0], set) or not
                isinstance(structures[0], dict)):
            return function(*structures)
        elif isinstance(structures[0], list):
            return [
                ExperienceReplay.nested_apply(
                    function,
                    *x,
                )
                for x in zip(*structures)
            ]
        elif isinstance(structures[0], tuple):
            return tuple(
                ExperienceReplay.nested_apply(
                    function,
                    *x,
                )
                for x in zip(*structures)
            )
        elif isinstance(structures[0], set):
            return {
                ExperienceReplay.nested_apply(
                    function,
                    *x,
                )
                for x in zip(*structures)
            }
        elif isinstance(structures[0], dict):
            keys_list = [structures[0].keys()]
            values_list = [y.values() for y in structures]
            merged_list = keys_list + values_list
            return {
                key: ExperienceReplay.nested_apply(
                    function,
                    *values,
                )
                for key, *values in zip(*merged_list)
            }

    def __init__(
        self, 
        env,
        policy: Policy,
    ):
        self.env = env
        self.policy = policy

    @abstractmethod
    def reset(
        self,
        max_size,
    ):
        return NotImplemented

    @abstractmethod
    def collect(
        self,
        max_path_length,
    ):
        return NotImplemented

    @abstractmethod
    def sample(
        self,
        batch_size,
    ):
        return NotImplemented
