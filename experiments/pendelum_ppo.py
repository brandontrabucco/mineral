"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.networks.policies.tanh_policy import TanhPolicy
from jetpack.networks.dense.dense_value_function import DenseValueFunction
from jetpack.envs.normalized_env import NormalizedEnv
from jetpack.data.path_buffer import PathBuffer
from jetpack.algorithms.actors.ppo import PPO
from jetpack.algorithms.critics.gae import GAE
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = TanhPolicy(
        GaussianPolicy(
            [32, 32, 2],
            lr=0.0001
        )
    )

    old_policy = TanhPolicy(
        GaussianPolicy(
            [32, 32, 2],
            lr=0.0001
        )
    )

    vf = DenseValueFunction(
        [6, 6, 1],
        lr=0.01
    )

    target_vf = DenseValueFunction(
        [6, 6, 1],
        lr=0.01
    )

    buffer = PathBuffer(
        env,
        policy
    )

    critic = GAE(
        vf,
        target_vf,
        gamma=1.0,
        lamb=1.0,
        monitor=monitor,
    )

    algorithm = PPO(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        epsilon=0.2,
        monitor=monitor
    )

    max_size = 32
    num_warm_up_paths = 32
    num_steps = 20000
    num_paths_to_collect = 32
    max_path_length = 100
    batch_size = 32
    num_trains_per_step = 1

    trainer = LocalTrainer(
        max_size,
        num_warm_up_paths,
        num_steps,
        num_paths_to_collect,
        max_path_length,
        batch_size,
        num_trains_per_step,
        buffer,
        algorithm,
        monitor=monitor
    )

    try:
        trainer.train()

    except KeyboardInterrupt:
        buffer.collect(1, save_paths=False, render=True)
