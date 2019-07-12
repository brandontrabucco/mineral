"""Author: Brandon Trabucco, Copyright 2019"""


import numpy as np
import jetpack as jp
from jetpack.buffers.buffer import Buffer


class PathBuffer(Buffer):

    def __init__(
        self,
        env,
        policy,
        selector=None,
    ):
        Buffer.__init__(self, env, policy)
        self.selector = (lambda x: x) if selector is None else selector
        self.max_size = None
        self.max_path_length = None
        self.size = None
        self.head = None
        self.tail = None
        self.observations = None
        self.actions = None
        self.rewards = None

    def reset(
        self,
        max_size,
        max_path_length
    ):
        self.max_size = max_size
        self.max_path_length = max_path_length
        self.size = 0
        self.head = 0
        self.tail = np.zeros([self.max_size], dtype=np.int32)

    def create(
        self,
        observation,
        action,
        reward
    ):
        def create_backend(x):
            return np.zeros([self.max_size, self.max_path_length, *x.shape], dtype=np.float32)

        self.observations = jp.nested_apply(create_backend, observation)
        self.actions = jp.nested_apply(create_backend, action)
        self.rewards = jp.nested_apply(create_backend, reward)

    def put(
        self,
        j,
        observation,
        action,
        reward
    ):
        def put_backend(x, y):
            x[self.head, j, ...] = y

        jp.nested_apply(put_backend, self.observations, observation)
        jp.nested_apply(put_backend, self.actions, action)
        jp.nested_apply(put_backend, self.rewards, reward)

    def collect(
        self,
        num_paths_to_collect,
        save_paths=True,
        render=False,
        **render_kwargs
    ):
        all_returns = []
        for i in range(num_paths_to_collect):
            observation = self.env.reset()
            path_return = 0.0
            for j in range(self.max_path_length):
                action = self.policy.get_stochastic_actions(
                    self.selector(observation)[np.newaxis, ...])[0, ...].numpy()
                next_observation, reward, done, info = self.env.step(action)
                if render:
                    self.env.render(**render_kwargs)
                if save_paths:
                    if self.size == 0:
                        self.create(observation, action, reward)
                    self.put(j, observation, action, reward)
                    self.tail[self.head] = j + 1
                path_return = path_return + reward
                if done:
                    break
                observation = next_observation
            if save_paths:
                self.head = (self.head + 1) % self.max_size
                self.size = min(self.size + 1, self.max_size)
            all_returns.append(path_return)
        return np.mean(all_returns)

    def sample(
        self,
        batch_size
    ):
        indices = np.random.choice(self.size, size=batch_size, replace=(self.size < batch_size))
        observations = jp.nested_apply(lambda x: x[indices, ...], self.observations)
        actions = jp.nested_apply(lambda x: x[indices, :(-1), ...], self.actions)
        rewards = jp.nested_apply(lambda x: x[indices, :(-1), ...], self.rewards)
        lengths = jp.nested_apply(lambda x: x[indices, ...], self.tail)
        max_lengths = np.arange(self.max_path_length)[np.newaxis, :]
        terminals = ((lengths[:, np.newaxis] - 1) > max_lengths).astype(np.float32)
        rewards = terminals[:, :(-1)] * rewards
        return (
            observations,
            actions,
            rewards,
            terminals
        )
