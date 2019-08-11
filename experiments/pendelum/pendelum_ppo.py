"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from mineral.networks.dense.dense_policy import DensePolicy
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.networks.dense.dense_value_function import DenseValueFunction
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.algorithms.actors.ppo import PPO
from mineral.algorithms.critics.gae import GAE
from mineral.core.trainers.local_trainer import LocalTrainer
from mineral.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [32, 32, 1],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    old_policy = DensePolicy(
        [32, 32, 1],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    vf = DenseValueFunction(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001}
    )

    target_vf = DenseValueFunction(
        [6, 6, 1],
        optimizer_kwargs={"lr": 0.0001}
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
