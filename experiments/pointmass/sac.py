"""Author: Brandon Trabucco, Copyright 2019"""


import multiprocessing
from mineral.precooked.sac import sac, sac_variant
from mineral.core.envs.debug.pointmass_env import PointmassEnv


def run_experiment(experiment_id):

    sac_variant["logging_dir"] = "./pointmass/sac/{}".format(experiment_id)
    sac_variant["reward_scale"] = 1.0
    sac_variant["hidden_size"] = 300
    sac_variant["tau"] = 0.005
    sac_variant["learning_rate"] = 0.0003
    sac_variant["batch_size"] = 10
    sac_variant["gamma"] = 0.99
    sac_variant["bellman_weight"] = 1.0
    sac_variant["discount_weight"] = 0.0
    sac_variant["max_size"] = 10000
    sac_variant["max_path_length"] = 10
    sac_variant["num_warm_up_paths"] = 100
    sac_variant["num_exploration_paths"] = 1
    sac_variant["num_evaluation_paths"] = 10
    sac_variant["num_threads"] = 16
    sac_variant["num_steps"] = 10000
    sac_variant["num_trains_per_step"] = 10

    sac(sac_variant, PointmassEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
