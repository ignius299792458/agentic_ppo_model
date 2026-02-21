import datetime
import random
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from src.agent import Agent
from src.args import parse_args
from src.env import make_envs


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    envs = make_envs(args, run_name)
    
    agent = Agent(envs)
    print("agent:", agent)
    print("agent.actor:", agent.actor)
    print("agent.critic:", agent.critic)
    
    # optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action spaces are supported"
    print("envs.single_action_space:", envs.single_action_space)
    print("envs.single_observation_space:", envs.single_observation_space)

    obs, info = envs.reset()
    for step in range(args.total_timesteps // args.num_envs):
        actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        obs, rewards, terminated, truncated, info = envs.step(actions)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
