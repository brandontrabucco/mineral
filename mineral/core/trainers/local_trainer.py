"""Author: Brandon Trabucco, Copyright 2019"""


import threading
from mineral.core.trainers.trainer import Trainer


class LocalTrainer(Trainer):

    def __init__(
        self,
        *args,
        num_steps=10000,
        num_trains_per_step=1,
        save_function=(lambda i: None),
        monitor=None,
    ):
        Trainer.__init__(
            self, 
            *args)
        self.num_steps = num_steps
        self.num_trains_per_step = num_trains_per_step
        self.save_function = save_function
        self.monitor = monitor

    def train(
        self
    ):
        best_reward = float("-inf")
        for iteration in range(self.num_steps):
            if iteration == 0:
                self.sampler.warm_up()

            expl_reward = self.sampler.explore()
            eval_reward = self.sampler.evaluate()
            if iteration > 0 and eval_reward > best_reward:
                best_reward = eval_reward
                self.save_function(iteration)

            print("CHECKING {}".format(self.monitor is not None))
            if self.monitor is not None:
                self.monitor.record("expl_average_reward", expl_reward)
                self.monitor.record("eval_average_reward", eval_reward)

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
