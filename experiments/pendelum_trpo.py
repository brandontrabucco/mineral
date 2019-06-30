"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.tanh_policy import TanhGaussianPolicy
from jetpack.networks.dense_value_function import DenseValueFunction
from jetpack.wrappers.normalized_env import NormalizedEnv
from jetpack.data.on_policy_buffer import OnPolicyBuffer
from jetpack.algorithms.trpo import TRPO
from jetpack.algorithms.critics.gae import GAE
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = TanhGaussianPolicy(
        [32, 32, 2],
        lr=0.01
    )

    old_policy = TanhGaussianPolicy(
        [32, 32, 2],
        lr=0.01
    )

    vf = DenseValueFunction(
        [6, 6, 1],
        lr=0.0001
    )

    buffer = OnPolicyBuffer(
        env,
        policy
    )

    critic = GAE(
        vf,
        gamma=1.0,
        lamb=1.0,
        monitor=monitor,
    )

    algorithm = TRPO(
        policy,
        old_policy,
        critic,
        gamma=0.99,
        delta=0.2,
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
        buffer.evaluate(1, render=True)
