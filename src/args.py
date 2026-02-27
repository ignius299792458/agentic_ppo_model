"""Command-line argument parsing for PPO hyperparameters.

Defines all configurable parameters: environment, training, PPO-specific
coefficients, logging, and rendering options. Also computes derived values
(batch_size, minibatch_size) from the provided arguments.
"""

import argparse
import os
from distutils.util import strtobool


def parse_args():
    """Parse CLI arguments and compute derived batch sizes.

    Derived values:
        batch_size = num_envs * num_steps
        minibatch_size = batch_size // num_minibatches

    Returns:
        argparse.Namespace: All hyperparameters as attributes.
    """
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
        "--num-steps",
        type=int,
        default=200,
        help="the number of steps to run in each batch",
    )

    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="the number of mini-batches",
    )

    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="the number of epochs to update the policy",
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
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, the learning rate will be annealed",
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
    
    parser.add_argument(
        "--renderAll",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
        help="if toggled, renders all environments.",
    )

    parser.add_argument(
        "--gae",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="if toggled, Generalized Advantage Estimation will be used",
        nargs="?",
        const=True,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor",
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda value for Generalized Advantage Estimation",
    )

    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the PPO clipping coefficient",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggles whether to clip the value loss",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="entropy coefficient",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="value function coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for gradient clipping",
    )

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args
