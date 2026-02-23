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
    args = parse_args()
    print(args)

    run_name = f"{args.gym_id}-{args.exp_name}-{args.seed}-{datetime.datetime.now().isoformat()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|params|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    seed_everything(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_envs(args, run_name)
    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
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
