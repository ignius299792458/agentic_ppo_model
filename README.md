# PPO Implementation

A clean, modular implementation of **Proximal Policy Optimization (PPO)** from scratch using PyTorch and Gymnasium.

## Setup

```bash
# Requires Python 3.11+
poetry install
```

## Usage

```bash
# Train with defaults (CartPole-v1)
python -m src.main

# Watch the agent live
python -m src.main --render

# Record videos
python -m src.main --capture-video

# Custom run
python -m src.main --gym-id CartPole-v1 --total-timesteps 100000 --num-envs 8

# View training metrics
tensorboard --logdir runs
```

## Project Structure

```
src/
├── main.py        Training loop (orchestration)
├── args.py        CLI argument parsing
├── agent.py       Actor-Critic network
├── env.py         Vectorized environment setup
├── storage.py     Rollout buffer allocation
├── rollout.py     Experience collection
├── advantage.py   GAE / returns computation
├── ppo.py         Minibatch clipped PPO update
├── logger.py      TensorBoard logging
├── utils.py       Seeding
└── docs/
    └── agent.md   In-depth architecture explanation
```

## Training Loop

```
for each update:
    1. rollout     → run policy in envs, collect (obs, actions, rewards, values)
    2. advantages  → compute GAE advantages and returns
    3. ppo_update  → shuffle into minibatches, run clipped PPO optimization
    4. log         → write metrics to TensorBoard
```

## Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--gym-id` | `CartPole-v1` | Gymnasium environment ID |
| `--total-timesteps` | `25000` | Total training steps |
| `--num-envs` | `8` | Parallel environments |
| `--num-steps` | `200` | Steps per rollout |
| `--num-minibatches` | `4` | Minibatches per update |
| `--update-epochs` | `4` | Epochs per PPO update |
| `--learning-rate` | `2.5e-4` | Adam learning rate |
| `--anneal-lr` | `True` | Linear LR annealing |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE lambda |
| `--clip-coef` | `0.2` | PPO clip range |
| `--ent-coef` | `0.01` | Entropy bonus weight |
| `--vf-coef` | `0.5` | Value loss weight |
| `--max-grad-norm` | `0.5` | Gradient clipping |
| `--render` | `False` | Live window |
| `--capture-video` | `False` | Save videos |
| `--track` | `False` | W&B logging |

## Dependencies

PyTorch, Gymnasium, NumPy, TensorBoard, pygame, moviepy, wandb
