"""Author: Brandon Trabucco, Copyright 2019"""


import gym
from jetpack.networks.dense_policy import DensePolicy
from jetpack.wrappers.proxy_env import ProxyEnv
from jetpack.data.on_policy_buffer import OnPolicyBuffer
from jetpack.algorithms.vpg import VPG
from jetpack.core.local_trainer import LocalTrainer
from jetpack.core.local_monitor import LocalMonitor


if __name__ == "__main__":

    monitor = LocalMonitor("./")

    env = ProxyEnv(
        gym.make("Pendulum-v0")
    )

    policy = DensePolicy(
        [32, 32, 1],
        lr=0.0001
    )

    buffer = OnPolicyBuffer(
        env,
        policy
    )

    algorithm = VPG(
        policy,
        gamma=0.99,
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
