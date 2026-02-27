"""Advantage and return computation.

Computes either Generalized Advantage Estimation (GAE) or simple
discounted returns from a completed rollout, using the critic's
value estimates and the collected rewards/dones.
"""

import torch


def compute_advantages(args, agent, next_obs, next_done, storage, device):
    """Compute advantages and returns from rollout data.

    GAE mode (args.gae=True):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        Returns = Advantages + Values

    Simple mode (args.gae=False):
        G_t = r_t + gamma * (1 - done_{t+1}) * G_{t+1}
        Advantages = Returns - Values

    All computation is done under torch.no_grad() since advantages
    are treated as fixed targets during the PPO update.

    Args:
        args: Parsed arguments with num_steps, gamma, gae, gae_lambda.
        agent: The Actor-Critic agent (used to bootstrap final value).
        next_obs: Observation after the last rollout step, shape (num_envs, obs_dim).
        next_done: Done flags after the last rollout step, shape (num_envs,).
        storage: Tuple of (obs, actions, logprobs, rewards, dones, values) tensors.
        device: torch.device for tensor placement.

    Returns:
        Tuple of (advantages, returns), each shape (num_steps, num_envs).
    """
    _, _, _, rewards, dones, values = storage

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    return advantages, returns
