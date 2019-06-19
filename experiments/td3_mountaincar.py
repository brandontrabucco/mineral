"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.dense_policy import DensePolicy
from jetpack.networks.dense_qf import DenseQF
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.data.off_policy_buffer import OffPolicyBuffer
from jetpack.algorithms.td3 import TD3
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = ProxyEnv(
        gym.make("MountainCarContinuous-v0")
    )

    policy = DensePolicy(
        [16, 16, 1]
    )

    qf1 = DenseQF(
        [32, 32, 1]
    )

    qf2 = DenseQF(
        [32, 32, 1]
    )

    target_policy = DensePolicy(
        [16, 16, 1]
    )

    target_qf1 = DenseQF(
        [32, 32, 1]
    )

    target_qf2 = DenseQF(
        [32, 32, 1]
    )

    buffer = OffPolicyBuffer(
        env,
        policy
    )

    algorithm = TD3(
        policy,
        qf1,
        qf2,
        target_policy,
        target_qf1,
        target_qf2,
        clip_radius=2.0,
        sigma=1.0,
        gamma=0.99,
        actor_delay=100,
        monitor=monitor
    )
    
    max_size = 1000
    num_warm_up_paths = 10
    num_steps = 1000
    num_paths_to_collect = 1
    max_path_length = 1000
    batch_size = 256
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
        buffer.evaluate(1, True, {})