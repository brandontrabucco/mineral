# Jetpack

A minimalist reinforcement learning package for TensorFlow 2.0.

## Features

Available algorithms for training policies.

- TRPO, PPO, SAC, TD3, DDPG, Actor Critic, Policy Gradient, Model Based RL
- Natural Gradient, Line Search, KL Constraint, KL Penalty

## Setup

Clone and install with pip.

```
git clone git@github.com:brandontrabucco/jetpack.git
pip install -e jetpack
```

## Training

Check progress by visiting http://localhost:6006.

```
python experiments/pendelum_trpo.py
```