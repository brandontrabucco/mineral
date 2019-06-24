"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.policies.dense_policy import DensePolicy
from jetpack.networks.dense_q_function import DenseQFunction
from jetpack.wrappers.normalized_env import NormalizedEnv
from jetpack.data.off_policy_buffer import OffPolicyBuffer
from jetpack.algorithms.critics.q_learning import QLearning
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = NormalizedEnv(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [6, 6, 1],
        lr=0.0001
    )

    qf = DenseQFunction(
        [6, 6, 1],
        lr=0.001
    )

    target_policy = DensePolicy(
        [6, 6, 1],
        lr=0.0001
    )

    target_qf = DenseQFunction(
        [6, 6, 1],
        lr=0.0001
    )

    buffer = OffPolicyBuffer(
        env,
        policy
    )

    algorithm = QLearning(
        qf,
        target_policy,
        target_qf,
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
