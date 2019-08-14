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
            for index in range(len(self.samplers)):
                if iteration == 0:
                    self.samplers[index].reset()
                    self.samplers[index].warm_up()

                exploration_return = self.samplers[index].explore()
                if self.monitor is not None:
                    self.monitor.record("exploration_return[{}]".format(
                        index), exploration_return)
                    evaluation_return = self.samplers[index].evaluate()
                    self.monitor.record("evaluation_return[{}]".format(
                        index), evaluation_return)

                for training_step in range(self.num_trains_per_step):
                    self.algorithms[index].gradient_update(
                        self.buffers[index])
