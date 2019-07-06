"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.gaussian_policy import GaussianPolicy
from jetpack.networks.policies.tanh_policy import TanhPolicy
from jetpack.networks.models.gaussian_model import GaussianModel
from jetpack.networks.models.tanh_model import TanhModel
from jetpack.envs.normalized_env import NormalizedEnv
from jetpack.data.path_buffer import PathBuffer
from jetpack.algorithms.transition_dynamics.one_step_regression import OneStepRegression
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
            lr=0.01
        )
    )

    model = TanhModel(
        GaussianModel(
            [32, 32, 6],
            lr=0.0001
        )
    )

    buffer = PathBuffer(
        env,
        policy
    )

    algorithm = OneStepRegression(
        model,
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
