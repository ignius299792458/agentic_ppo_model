"""Rollout buffer allocation.

Pre-allocates GPU/CPU tensors to store one rollout's worth of data:
observations, actions, log-probabilities, rewards, done flags, and values.
Shape: (num_steps, num_envs, ...) for each tensor.
"""

import torch


def init_storage(args, envs, device):
    """Allocate zero-filled tensors for storing rollout data.

    Args:
        args: Parsed arguments with num_steps, num_envs.
        envs: Vectorized environment (used to read obs/action shapes).
        device: torch.device to place tensors on.

    Returns:
        Tuple of (obs, actions, logprobs, rewards, dones, values),
        each a tensor of shape (num_steps, num_envs, ...).
    """
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    return obs, actions, logprobs, rewards, dones, values
