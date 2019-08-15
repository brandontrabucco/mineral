"""Author: Brandon Trabucco, Copyright 2019"""


from mineral.core.trainers.trainer import Trainer


class LocalTrainer(Trainer):

    def __init__(
        self,
        *args,
        num_steps=10000,
        num_trains_per_step=1,
        monitor=None
    ):
        Trainer.__init__(
            self, 
            *args)
        self.num_steps = num_steps
        self.num_trains_per_step = num_trains_per_step
        self.monitor = monitor

    def train(
        self
    ):
        for iteration in range(self.num_steps):
            if iteration == 0:
                self.sampler.reset()
                self.sampler.warm_up()

            exploration_return = self.sampler.explore()
            if self.monitor is not None:
                self.monitor.record("exploration_return", exploration_return)
                evaluation_return = self.sampler.evaluate()
                self.monitor.record("evaluation_return", evaluation_return)

            for index in range(len(self.algorithms)):
                for training_step in range(self.num_trains_per_step):
                    self.algorithms[index].gradient_update(
                        self.buffers[index])
