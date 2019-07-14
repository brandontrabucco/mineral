"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.core.trainers.trainer import Trainer


class LocalTrainer(Trainer):

    def __init__(
        self,
        max_size,
        num_warm_up_paths,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        buffer,
        algorithm,
        monitor=None
    ):
        Trainer.__init__(
            self, 
            buffer, 
            algorithm
        )
        self.max_size = max_size
        self.num_warm_up_paths = num_warm_up_paths
        self.num_steps = num_steps
        self.num_paths_to_collect = num_paths_to_collect
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.num_trains_per_step = num_trains_per_step
        self.monitor = monitor

    def train(
        self
    ):
        self.buffer.reset(self.max_size, self.max_path_length)
        self.buffer.collect(self.num_warm_up_paths, random=True, save_paths=True)

        for i in range(self.num_steps):
            expl_r = self.buffer.collect(self.num_paths_to_collect, random=True, save_paths=True)
            if self.monitor is not None:
                self.monitor.record("exploration_return", expl_r)

            eval_r = self.buffer.collect(self.num_paths_to_collect, random=False, save_paths=False)
            if self.monitor is not None:
                self.monitor.record("evaluation_return", eval_r)

            for j in range(self.num_trains_per_step):
                batch = self.buffer.sample(self.batch_size)
                self.algorithm.gradient_update(*batch)