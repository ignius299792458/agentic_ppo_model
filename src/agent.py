"""Actor-Critic neural network agent for PPO.

Implements a two-headed network:
  - Actor: outputs action logits -> Categorical distribution over discrete actions
  - Critic: outputs a scalar state-value estimate V(s)

Both heads use orthogonal initialization with Tanh activations.
"""

import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a linear layer with orthogonal weights and constant bias.

    Orthogonal initialization preserves gradient magnitude across layers,
    preventing vanishing/exploding gradients. The std scales the weight matrix.

    Args:
        layer: An nn.Linear layer to initialize.
        std: Scale factor for orthogonal weights.
             sqrt(2) for hidden layers (compensates Tanh),
             0.01 for actor output (near-uniform initial policy),
             1.0 for critic output (standard scale).
        bias_const: Constant value for all biases (default 0).

    Returns:
        The initialized layer (in-place, also returned for inline use).
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-Critic agent with separate policy and value networks.

    Architecture (both heads):
        Linear(obs_dim, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, out)

    Actor output: n_actions logits (fed into Categorical distribution)
    Critic output: 1 scalar (estimated state value)

    Args:
        envs: Vectorized gymnasium environment (used to read obs/action shapes).
    """

    def __init__(self, envs):
        super(Agent, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        """Estimate the value of a state (critic only).

        Args:
            x: Observation tensor of shape (batch, obs_dim).

        Returns:
            Tensor of shape (batch, 1) with V(s) estimates.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Run both actor and critic on the observation.

        During rollout (action=None): samples a new action from the policy.
        During PPO update (action provided): evaluates the given action.

        Args:
            x: Observation tensor of shape (batch, obs_dim).
            action: Optional pre-selected action tensor. If None, samples one.

        Returns:
            Tuple of (action, log_prob, entropy, value):
                action: Sampled or provided action, shape (batch,).
                log_prob: log pi(action|obs), shape (batch,).
                entropy: Policy entropy H(pi), shape (batch,).
                value: V(s) estimate, shape (batch, 1).
        """
        action_logits = self.actor(x)
        distribution = Categorical(logits=action_logits)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic(x),
        )
