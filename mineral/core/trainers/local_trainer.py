"""Author: Brandon Trabucco, Copyright 2019"""


import threading
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
                self.monitor.record("evaluation_return", self.sampler.evaluate())

            def inner_train(algorithm, buffer, num_trains):
                for training_step in range(num_trains):
                    algorithm.gradient_update(buffer)

            threads = [threading.Thread(
                target=inner_train, args=(a, b, self.num_trains_per_step))
                for a, b in zip(self.algorithms, self.buffers)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()
