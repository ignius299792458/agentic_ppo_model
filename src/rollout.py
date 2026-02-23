import numpy as np
import torch


def rollout(args, agent, envs, device, storage, next_obs, next_done, global_step, writer):
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
