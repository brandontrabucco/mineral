"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.algorithms.base import Base
from jetpack.data.experience_replay import ExperienceReplay


class Trainer(object):

    def __init__(
        self,
        replay: ExperienceReplay,
        algorithm: Base,
    ):
        self.replay = replay
        self.algorithm = algorithm

    def train(
        self,
        max_size,
        num_steps,
        max_path_length,
        batch_size,
    ):
        self.replay.reset(max_size)
        for i in range(num_steps):
            self.replay.collect(max_path_length)
            batch = self.replay.sample(batch_size)
            self.algorithm.gradient_update(*batch)