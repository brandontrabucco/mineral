"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.dense.dense_policy import DensePolicy
from jetpack.distributions.tanh_gaussian_distribution import TanhGaussianDistribution
from jetpack.networks.dense.dense_forward_model import DenseForwardModel
from jetpack.core.environments.normalized_environment import NormalizedEnvironment
from jetpack.buffers.path_buffer import PathBuffer
from jetpack.algorithms.transition_dynamics.one_step_prediction import OneStepPrediction
from jetpack.core.trainers.local_trainer import LocalTrainer
from jetpack.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnvironment(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [32, 32, 2],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    model = DenseForwardModel(
        [32, 32, 6],
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
