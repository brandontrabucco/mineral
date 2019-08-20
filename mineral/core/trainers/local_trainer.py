"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
from mineral.core.trainers.trainer import Trainer


class LocalTrainer(Trainer):

    def __init__(
        self,
        *args,
        num_steps=10000,
        num_trains_per_step=1,
        saver=None,
        monitor=None,
    ):
        Trainer.__init__(
            self, 
            *args)
        self.num_steps = num_steps
        self.num_trains_per_step = num_trains_per_step
        self.saver = saver
        self.monitor = monitor

    def train(
        self
    ):
        best_reward = float("-inf")
        for iteration in range(self.num_steps):
            if iteration == 0:
                self.sampler.warm_up()

            expl_reward = np.mean(self.sampler.explore())
            eval_reward = np.mean(self.sampler.evaluate())
            if iteration > 0 and eval_reward > best_reward:
                best_reward = eval_reward
                if self.saver is not None:
                    self.saver.save(iteration)

            if self.monitor is not None:
                self.monitor.record("expl_average_reward", expl_reward)
                self.monitor.record("eval_average_reward", eval_reward)

            for a, b in zip(self.algorithms, self.buffers):
                for training_step in range(self.num_trains_per_step):
                    a.gradient_update(b)
