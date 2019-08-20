"""Author: Brandon Trabucco, Copyright 2019"""


import multiprocessing
from mineral.baselines.td3 import td3, td3_variant
from mineral.core.envs.debug.pointmass_env import PointmassEnv


def run_experiment(experiment_id):

    td3_variant["logging_dir"] = "./pointmass/td3/{}".format(experiment_id)
    td3_variant["reward_scale"] = 1.0
    td3_variant["hidden_size"] = 300
    td3_variant["tau"] = 0.005
    td3_variant["learning_rate"] = 0.0003
    td3_variant["batch_size"] = 128
    td3_variant["gamma"] = 0.99
    td3_variant["bellman_weight"] = 1.0
    td3_variant["discount_weight"] = 0.0
    td3_variant["max_size"] = 10000
    td3_variant["max_path_length"] = 10
    td3_variant["num_warm_up_paths"] = 1000
    td3_variant["num_exploration_paths"] = 1
    td3_variant["num_evaluation_paths"] = 10
    td3_variant["num_threads"] = 10
    td3_variant["num_steps"] = 10000
    td3_variant["num_trains_per_step"] = 10

    td3(td3_variant, PointmassEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
