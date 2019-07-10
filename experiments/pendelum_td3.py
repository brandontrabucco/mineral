"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.dense.dense_policy import DensePolicy
from jetpack.distributions.tanh_gaussian_distribution import TanhGaussianDistribution
from jetpack.networks.dense.dense_q_function import DenseQFunction
from jetpack.core.envs.normalized_env import NormalizedEnv
from jetpack.buffers.path_buffer import PathBuffer
from jetpack.algorithms.actors.ddpg import DDPG
from jetpack.algorithms.critics.q_learning import QLearning
from jetpack.algorithms.critics.twin_delayed_critic import TwinDelayedCritic
from jetpack.core.trainers.local_trainer import LocalTrainer
from jetpack.core.monitors.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [32, 32, 2],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    qf1 = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
    )

    qf2 = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
    )

    target_policy = DensePolicy(
        [32, 32, 2],
        optimizer_kwargs={"lr": 0.0001},
        distribution_class=TanhGaussianDistribution
    )

    target_qf1 = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
    )

    target_qf2 = DenseQFunction(
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
    )

    buffer = PathBuffer(
        env,
        policy
    )

    clip_radius = 0.2
    std = 0.1
    gamma = 0.99
    actor_delay = 10

    q_backup1 = QLearning(
        target_policy,
        qf1,
        target_qf1,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        monitor=monitor,
    )

    q_backup2 = QLearning(
        target_policy,
        qf2,
        target_qf2,
        gamma=gamma,
        clip_radius=clip_radius,
        std=std,
        monitor=monitor,
    )

    twin_delayed_q_backup = TwinDelayedCritic(
        q_backup1,
        q_backup2
    )

    algorithm = DDPG(
        policy,
        twin_delayed_q_backup,
        target_policy,
        actor_delay=actor_delay,
        monitor=None,
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
        buffer.collect(1, save_paths=False, render=True)
