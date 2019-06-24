"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.dense_policy import DensePolicy
from jetpack.networks.dense_q_function import DenseQFunction
from jetpack.wrappers.normalized_env import NormalizedEnv
from jetpack.data.off_policy_buffer import OffPolicyBuffer
from jetpack.algorithms.ddpg import DDPG
from jetpack.algorithms.critics.q_learning import QLearning
from jetpack.algorithms.critics.twin_delayed_q_learning import TwinDelayedQLearning
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
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
        [6, 6, 1],
        tau=1e-2,
        lr=0.0001
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

    buffer = OffPolicyBuffer(
        env,
        policy
    )

    clip_radius = 0.2
    sigma = 0.1
    gamma = 0.99
    actor_delay = 10

    q_backup1 = QLearning(
        qf1,
        target_policy,
        target_qf1,
        gamma=gamma,
        clip_radius=clip_radius,
        sigma=sigma,
        monitor=monitor,
    )

    q_backup2 = QLearning(
        qf2,
        target_policy,
        target_qf2,
        gamma=gamma,
        clip_radius=clip_radius,
        sigma=sigma,
        monitor=monitor,
    )

    twin_delayed_q_backup = TwinDelayedQLearning(
        q_backup1,
        q_backup2,
        monitor=monitor,
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
        buffer.evaluate(1, render=True)