"""Author: Brandon Trabucco, Copyright 2019"""


import multiprocessing
from mineral.baselines.monolithic.pg import pg, pg_variant
from mineral.core.envs.debug.pointmass_env import PointmassEnv


def run_experiment(experiment_id):

    pg_variant["logging_dir"] = "./pointmass/pg/{}".format(experiment_id)
    pg_variant["reward_scale"] = 1.0
    pg_variant["hidden_size"] = 300
    pg_variant["tau"] = 0.005
    pg_variant["learning_rate"] = 0.0003
    pg_variant["batch_size"] = 32
    pg_variant["gamma"] = 0.99
    pg_variant["bellman_weight"] = 0.0
    pg_variant["discount_weight"] = 1.0
    pg_variant["max_size"] = 32
    pg_variant["max_path_length"] = 10
    pg_variant["num_warm_up_paths"] = 0
    pg_variant["num_exploration_paths"] = 32
    pg_variant["num_evaluation_paths"] = 32
    pg_variant["num_threads"] = 16
    pg_variant["num_steps"] = 10000
    pg_variant["num_trains_per_step"] = 1

    pg(pg_variant, PointmassEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
