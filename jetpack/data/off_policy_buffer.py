"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from jetpack.data.path_buffer import PathBuffer


class OffPolicyBuffer(PathBuffer):
                
    def sample(
        self,
        batch_size
    ):
        indices = np.random.choice(
            self.candidates.shape[0],
            size=batch_size, 
            replace=(self.candidates.shape[0] < batch_size)
        )
        path_ind = self.candidates[indices, 0]
        step_ind = self.candidates[indices, 1]
        select = lambda x: x[path_ind, step_ind, ...]
        select_next = lambda x: x[path_ind, step_ind + 1, ...]
        return (
            jp.nested_apply(
                select,
                self.observations
            ),
            jp.nested_apply(
                select,
                self.actions
            ),
            jp.nested_apply(
                select,
                self.rewards
            ),
            jp.nested_apply(
                select_next,
                self.observations
            ),
        )
