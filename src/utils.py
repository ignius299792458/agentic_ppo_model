"""Utility functions for reproducibility and setup."""

import random
import numpy as np
import torch


def seed_everything(args):
    """Set random seeds across all libraries for reproducible results.

    Seeds Python's random, NumPy, and PyTorch RNGs. Optionally forces
    cuDNN to use deterministic algorithms (slower but reproducible).

    Args:
        args: Parsed arguments with `seed` (int) and `torch_deterministic` (bool).
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
