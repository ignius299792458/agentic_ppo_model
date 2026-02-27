"""PPO training entry point.

Orchestrates the full training loop:
  1. Parse args, set seeds, create envs and agent
  2. For each update:
     a. Optionally anneal learning rate
     b. Collect rollout experience
     c. Compute advantages (GAE)
     d. Run PPO minibatch updates
     e. Log metrics to TensorBoard
  3. Clean up

Usage:
    python -m src.main
    python -m src.main --render --total-timesteps 100000
"""

import datetime
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from src.agent import Agent
from src.args import parse_args
from src.env import make_envs
from src.storage import init_storage
from src.rollout import rollout
from src.advantage import compute_advantages
from src.ppo import ppo_update
from src.logger import log_metrics
from src.utils import seed_everything


def main():
    """Run the full PPO training loop.

    Flow:
        parse_args -> seed -> make_envs -> Agent -> init_storage
        -> for each update: rollout -> compute_advantages -> ppo_update -> log
        -> close envs and writer
    """
    args = parse_args()
    print("ARGS: ", args, "\n")

    run_name = f"{args.gym_id}-{args.exp_name}-{args.seed}-{datetime.datetime.now().isoformat()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|params|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    # Seed everything
    seed_everything(args)
    
    # set devive if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # make parallel sync vector envs
    envs = make_envs(args, run_name)
    
    # create agent
    agent = Agent(envs).to(device)
    
    # create optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # initialize storage
    storage = init_storage(args, envs, device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        next_obs, next_done, global_step = rollout(
            args, agent, envs, device, storage, next_obs, next_done, global_step, writer
        )

        advantages, returns = compute_advantages(
            args, agent, next_obs, next_done, storage, device
        )

        metrics = ppo_update(args, agent, optimizer, envs, storage, advantages, returns)

        sps = int(global_step / (time.time() - start_time))
        log_metrics(writer, optimizer, metrics, global_step, sps, update, num_updates)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
