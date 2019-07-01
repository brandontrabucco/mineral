"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.mean_gaussian_policy import MeanGaussianPolicy
from jetpack.networks.dense.dense_value_function import DenseValueFunction
from jetpack.wrappers.normalized_env import NormalizedEnv
from jetpack.data.on_policy_buffer import OnPolicyBuffer
from jetpack.algorithms.critics.value_regression import ValueRegression
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = MeanGaussianPolicy(
        [6, 6, 1],
        lr=0.0001
    )

    vf = DenseValueFunction(
        [6, 6, 1],
        lr=0.0001
    )

    buffer = OnPolicyBuffer(
        env,
        policy
    )

    algorithm = ValueRegression(
        vf,
        gamma=0.99,
        monitor=monitor
    )
    
    max_size = 1000
    num_warm_up_paths = 10
    num_steps = 20000
    num_paths_to_collect = 1
    max_path_length = 100
    batch_size = 32
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
        buffer.evaluate(1, render=True)
