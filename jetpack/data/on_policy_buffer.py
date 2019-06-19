"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from jetpack.data.path_buffer import PathBuffer


class OnPolicyBuffer(PathBuffer):
                
    def sample(
        self,
        batch_size
    ):
        indices = np.random.choice(
            self.size,
            size=batch_size, 
            replace=(self.size < batch_size)
        )
        select_minus_one = lambda x: x[indices, :(-1), ...]
        select = lambda x: x[indices, ...]
        return (
            jp.nested_apply(
                select,
                self.observations
            ),
            jp.nested_apply(
                select_minus_one,
                self.actions
            ),
            jp.nested_apply(
                select_minus_one,
                self.rewards
            ),
            jp.nested_apply(
                select,
                self.tail
            )
        )
