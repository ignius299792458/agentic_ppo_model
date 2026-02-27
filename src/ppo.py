"""PPO clipped minibatch optimization.

Implements the core PPO update: flattens the rollout batch, shuffles into
minibatches, and performs multiple epochs of clipped policy and value
updates with entropy regularization.
"""

import numpy as np
import torch
import torch.nn as nn


def ppo_update(args, agent, optimizer, envs, storage, advantages, returns):
    """Perform the PPO clipped surrogate objective update.

    Steps:
      1. Flatten rollout from (num_steps, num_envs) to (batch_size,)
      2. For each epoch, shuffle indices and split into minibatches
      3. For each minibatch:
         a. Recompute log_prob, entropy, value from current policy
         b. Compute probability ratio: r = exp(new_logprob - old_logprob)
         c. Clipped policy loss: max(-A*r, -A*clip(r, 1-eps, 1+eps))
         d. Clipped value loss: 0.5 * max((V-R)^2, (V_clip-R)^2)
         e. Total loss = policy_loss + vf_coef*value_loss - ent_coef*entropy
         f. Backprop with gradient clipping

    Args:
        args: Parsed arguments with batch_size, minibatch_size, update_epochs,
              clip_coef, clip_vloss, ent_coef, vf_coef, max_grad_norm.
        agent: The Actor-Critic agent.
        optimizer: torch.optim.Adam optimizer.
        envs: Vectorized environment (used to read obs/action shapes).
        storage: Tuple of (obs, actions, logprobs, rewards, dones, values) tensors.
        advantages: Advantage estimates, shape (num_steps, num_envs).
        returns: Computed returns, shape (num_steps, num_envs).

    Returns:
        Dict with loss metrics: value_loss, policy_loss, entropy,
        old_approx_kl, approx_kl, clipfrac, explained_variance.
    """
    obs, actions, logprobs, _, _, values = storage

    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    b_inds = np.arange(args.batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                b_obs[mb_inds], b_actions.long()[mb_inds]
            )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                )

            mb_advantages = b_advantages[mb_inds]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "value_loss": v_loss.item(),
        "policy_loss": pg_loss.item(),
        "entropy": entropy_loss.item(),
        "old_approx_kl": old_approx_kl.item(),
        "approx_kl": approx_kl.item(),
        "clipfrac": np.mean(clipfracs),
        "explained_variance": explained_var,
    }
