"""Author: Brandon Trabucco, Copyright 2019"""


from jetpack.core.trainer import Trainer
from jetpack.algorithms.base import Base
from jetpack.data.buffer import Buffer


class BatchTrainer(Trainer):

    def __init__(
        self,
        max_size,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        buffer: Buffer,
        algorithm: Base,
    ):
        Trainer.__init__(
            self, 
            buffer, 
            algorithm,
        )
        self.max_size = max_size
        self.num_steps = num_steps
        self.num_paths_to_collect = num_paths_to_collect
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.num_trains_per_step = num_trains_per_step

    def train(
        self,
    ):
        self.buffer.reset(self.max_size)
        for i in range(self.num_steps):
            self.buffer.collect(
                self.num_paths_to_collect,
                self.max_path_length,
            )
            for j in range(self.num_trains_per_step):
                batch = self.buffer.sample(self.batch_size)
                self.algorithm.gradient_update(*batch)