"""Experience collection (rollout phase).

Runs the current policy in the vectorized environment for num_steps,
storing observations, actions, log-probs, rewards, dones, and values
into the pre-allocated storage tensors. Logs episode statistics when
environments finish episodes.
"""

import numpy as np
import torch


def rollout(args, agent, envs, device, storage, next_obs, next_done, global_step, writer):
    """Collect a full rollout of experience from the environment.

    For each step:
      1. Query agent for action, log_prob, value (no gradient)
      2. Step the environment with the action
      3. Store transition data in the storage buffers
      4. Log completed episode stats to TensorBoard

    Args:
        args: Parsed arguments with num_steps, num_envs.
        agent: The Actor-Critic agent.
        envs: Vectorized gymnasium environment.
        device: torch.device for tensor placement.
        storage: Tuple of (obs, actions, logprobs, rewards, dones, values) tensors.
        next_obs: Current observation tensor, shape (num_envs, obs_dim).
        next_done: Current done flags tensor, shape (num_envs,).
        global_step: Running count of total environment steps taken.
        writer: TensorBoard SummaryWriter for logging episode returns.

    Returns:
        Tuple of (next_obs, next_done, global_step) — updated state after rollout.
    """
    obs, actions, logprobs, rewards, dones, values = storage

    for step in range(args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        rewards[step] = torch.Tensor(reward).to(device).view(-1)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.Tensor(done).to(device)

        if "_episode" in info:
            for i, done_flag in enumerate(info["_episode"]):
                if done_flag:
                    ep_return = info["episode"]["r"][i]
                    ep_length = info["episode"]["l"][i]
                    print(f"global_step={global_step}, episodic_return={ep_return}")
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_length, global_step)
                    break

    return next_obs, next_done, global_step
