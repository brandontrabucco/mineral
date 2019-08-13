"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.core.trainers.trainer import Trainer


class LocalTrainer(Trainer):

    def __init__(
        self,
        *inputs,
        num_warm_up_paths=32,
        num_steps=1000,
        num_paths_to_collect=32,
        batch_size=32,
        num_trains_per_step=1,
        monitor=None
    ):
        Trainer.__init__(
            self, 
            *inputs
        )
        self.num_warm_up_paths = num_warm_up_paths
        self.num_steps = num_steps
        self.num_paths_to_collect = num_paths_to_collect
        self.batch_size = batch_size
        self.num_trains_per_step = num_trains_per_step
        self.monitor = monitor

    def train(
        self
    ):
        for buffer in self.buffers:
            buffer.reset()
            buffer.collect(self.num_warm_up_paths, random=True, save_paths=True)

        for i in range(self.num_steps):
            for b in range(len(self.buffers)):
                expl_r = self.buffers[b].collect(self.num_paths_to_collect, random=True, save_paths=True)
                eval_r = self.buffers[b].collect(self.num_paths_to_collect, random=False, save_paths=False)

                for j in range(self.num_trains_per_step):
                    batch = self.buffers[b].sample(self.batch_size)
                    self.algorithms[b].gradient_update(*batch)

                if self.monitor is not None:
                    self.monitor.record("exploration_return_{}".format(b), expl_r)
                    self.monitor.record("evaluation_return_{}".format(b), eval_r)