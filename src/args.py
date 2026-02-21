import argparse
import os
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--gym-id",
        type=str,
        default="CartPole-v1",
        help="the id of the gym environment",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="the seed of the experiment"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="the number of environments to run in parallel",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=25000,
        help="total timesteps of the training",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="if toggled, cuda will be enabled by default",
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="if toggled, we will track the model and hyperparameter performance with wandb",
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="the wandb project name",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="the wandb entity name"
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        default=False,
        help="if toggled, videos will be saved of the agent playing the game.",
    )
    parser.add_argument(
        "--render",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
        help="if toggled, opens a live window to watch the agent play.",
    )
    args = parser.parse_args()
    # args.batch_size = int(args. * args.num_steps)
    # args.minibatch_size = int(args.batch_size // 4)
    return args