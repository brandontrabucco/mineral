"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from mineral.networks.dense.dense_policy import DensePolicy
from mineral.distributions.gaussians.tanh_gaussian_distribution import TanhGaussianDistribution
from mineral.networks.dense.dense_forward_model import DenseForwardModel
from mineral.core.envs.normalized_env import NormalizedEnv
from mineral.buffers.path_buffer import PathBuffer
from mineral.algorithms.dynamics_models.one_step_prediction import OneStepPrediction
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

    model = DenseForwardModel(
        [32, 32, 3],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    buffer = PathBuffer(
        env,
        policy
    )

    algorithm = OneStepPrediction(
        model,
        monitor=monitor
    )

    max_size = 32
    num_warm_up_paths = 32
    num_steps = 20000
    num_paths_to_collect = 1
    max_path_length = 100
    batch_size = 8
    num_trains_per_step = 100

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
