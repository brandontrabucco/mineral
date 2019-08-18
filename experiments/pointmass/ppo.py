"""Author: Brandon Trabucco, Copyright 2019"""


import multiprocessing
from mineral.baselines.ppo import ppo, ppo_variant
from mineral.core.envs.debug.pointmass_env import PointmassEnv


def run_experiment(experiment_id):

    ppo_variant["logging_dir"] = "./pointmass/ppo/{}".format(experiment_id)
    ppo_variant["reward_scale"] = 1.0
    ppo_variant["hidden_size"] = 300
    ppo_variant["tau"] = 0.005
    ppo_variant["learning_rate"] = 0.0003
    ppo_variant["batch_size"] = 32
    ppo_variant["gamma"] = 0.99
    ppo_variant["lamb"] = 0.95
    ppo_variant["epsilon"] = 0.1
    ppo_variant["bellman_weight"] = 0.0
    ppo_variant["discount_weight"] = 1.0
    ppo_variant["max_size"] = 32
    ppo_variant["max_path_length"] = 10
    ppo_variant["num_warm_up_paths"] = 0
    ppo_variant["num_exploration_paths"] = 32
    ppo_variant["num_evaluation_paths"] = 32
    ppo_variant["num_threads"] = 16
    ppo_variant["num_steps"] = 10000
    ppo_variant["num_trains_per_step"] = 10

    ppo(ppo_variant, PointmassEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
