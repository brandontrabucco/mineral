# mineral

A minimalist reinforcement learning package for TensorFlow 2.0. Have fun! -Brandon

## Features

Available algorithms for training policies.

- TRPO, PPO, SAC, TD3, DDPG, Actor Critic, Policy Gradient, Model Based RL
- Natural Gradient, Line Search, KL Constraint, KL Penalty

## Setup

Clone and install with pip.

```console 
git clone git@github.com:brandontrabucco/mineral.git
pip install -e mineral
```

## Training

Launch experiments, then check progress at http://localhost:6006.

```console 
python experiments/pendelum_trpo.py
```

## Designing Experiments

Experiments are launched by calling the **train** method.

```python
trainer.train()
```

The **trainer** performs gradient updates.

```python
trainer = LocalTrainer(
    max_num_paths,
    num_warm_up_paths,
    num_gradient_updates,
    num_paths_to_collect,
    max_path_length,
    batch_size,
    num_trains_per_step,
    buffer,
    algorithm
)
```

Several **algorithms** implement gradient updates.

```python
critic = GAE(
    vf,
    target_vf
)

algorithm = PPO(
    policy,
    critic
)
```

A **buffer** stores transitions for training.

```python
buffer = PathBuffer(
    env,
    policy
)
```

Several **neural networks** shall be trained.

```python
policy = DensePolicy(
    [hidden_size, hidden_size, action_size]
)

vf = DenseValueFunction(
    [hidden_size, hidden_size, 1]
)

target_vf = DenseValueFunction(
    [hidden_size, hidden_size, 1]
)
```

An **environment** samples transitions for training.

```python
env = NormalizedEnvironment(
    gym.make(env_name)
)
```
