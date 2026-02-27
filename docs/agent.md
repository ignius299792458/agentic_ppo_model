# From Zero to PPO: Understanding `src/agent.py`

This document teaches you everything you need to understand the agent file,
starting from the absolute basics.

---

# Part 0: The Imports — What Each Line Brings In

```python
import torch.nn as nn
import torch.distributions.categorical as Categorical
import numpy as np
```

Let's understand each one:

---

## `import torch.nn as nn`

`torch` is **PyTorch** — the deep learning framework. Think of it as a toolbox
for building and training neural networks.

`torch.nn` is the **neural network** sub-module. We alias it as `nn` for short.

It gives us everything we need to build a network:

```
nn.Module         The base class for ALL neural networks.
                  Our Agent inherits from it.
                  It automatically tracks all weights inside the network.

nn.Sequential     A container that chains layers together.
                  Data flows through them in order: layer1 → layer2 → layer3

nn.Linear         A single "fully connected" layer.
                  Does the math: output = weights × input + bias
                  nn.Linear(4, 64) = 4 inputs, 64 outputs
                  This layer has 4×64 = 256 weights + 64 biases = 320 learnable numbers

nn.Tanh           Activation function. Squashes any number to [-1, +1].
                  Adds non-linearity so the network can learn complex patterns.

nn.init           Sub-module for weight initialization.
                  nn.init.orthogonal_()  → set weights to orthogonal matrix
                  nn.init.constant_()    → set values to a constant
```

### How They Compose Together

```python
nn.Sequential(
    nn.Linear(4, 64),   # Layer 1: 4 → 64 numbers
    nn.Tanh(),           # Squash to [-1, +1]
    nn.Linear(64, 64),   # Layer 2: 64 → 64 numbers
    nn.Tanh(),           # Squash again
    nn.Linear(64, 2),    # Layer 3: 64 → 2 numbers (one per action)
)
```

```
Input [4]                     ← observation from CartPole
   │
   ▼
nn.Linear(4, 64)             ← 320 learnable parameters (4×64 weights + 64 biases)
   │  output = W₁ × input + b₁
   ▼
nn.Tanh()                    ← no parameters, just math: tanh(x)
   │
   ▼
nn.Linear(64, 64)            ← 4160 learnable parameters (64×64 + 64)
   │  output = W₂ × input + b₂
   ▼
nn.Tanh()
   │
   ▼
nn.Linear(64, 2)             ← 130 learnable parameters (64×2 + 2)
   │  output = W₃ × input + b₃
   ▼
Output [2]                    ← one score per action (left, right)

Total learnable parameters: 320 + 4160 + 130 = 4610
These 4610 numbers are what "learning" adjusts!
```

---

## `import torch.distributions.categorical as Categorical`

This gives us the **Categorical distribution** — a probability distribution
over a finite set of choices (like picking from a menu).

```
What is a Categorical Distribution?

Imagine rolling a loaded die:
  Face 1: 10% chance
  Face 2: 30% chance
  Face 3: 60% chance

That's a Categorical distribution over {1, 2, 3}.

For CartPole with 2 actions:
  LEFT:  85% chance
  RIGHT: 15% chance

That's a Categorical distribution over {LEFT, RIGHT}.
```

### How It's Used in the Agent

```python
logits = actor_network(observation)        # → [1.5, -0.3]
distribution = Categorical(logits=logits)  # converts to probabilities internally
```

```
What are "logits"?

Logits are RAW scores — they can be any number.
Categorical converts them to probabilities using softmax:

  logits:        [1.5,  -0.3]
                   ↓
  softmax:       [e^1.5 / (e^1.5 + e^-0.3),  e^-0.3 / (e^1.5 + e^-0.3)]
                 [4.48 / 5.22,                 0.74 / 5.22]
                   ↓
  probabilities: [0.86,  0.14]
                  LEFT   RIGHT
```

### Key Methods

```
distribution.sample()            Pick a random action based on probabilities.
                                 86% of the time → LEFT, 14% → RIGHT.

distribution.log_prob(action)    "How likely was this action?"
                                 Returns log(probability).
                                 log(0.86) = -0.15 for LEFT
                                 log(0.14) = -1.97 for RIGHT
                                 Used in PPO's ratio calculation.

distribution.entropy()           "How uncertain is the distribution?"
                                 High entropy [0.5, 0.5] = very uncertain
                                 Low entropy [0.99, 0.01] = very confident
                                 PPO uses this to encourage exploration.
```

### Why log_prob Instead of Just prob?

```
Math reasons:
1. Probabilities multiply: P(A and B) = P(A) × P(B)
   Logs turn multiplication into addition: log(A×B) = log(A) + log(B)
   Addition is faster and more numerically stable.

2. Probabilities can be tiny: 0.000001
   log(0.000001) = -13.8  ← much easier for computers to work with

3. PPO needs the RATIO of probabilities:
   ratio = new_prob / old_prob
   = exp(new_log_prob - old_log_prob)    ← subtraction instead of division!
```

---

## `import numpy as np`

**NumPy** is Python's library for fast math with arrays of numbers.

In the agent, it's used for two things:

```python
np.sqrt(2)
# → 1.4142...
# The default std for layer initialization.
# Why √2? It compensates for Tanh activation squashing values.

np.prod(envs.single_observation_space.shape)
# → 4 (for CartPole)
# Multiplies all dimensions of the observation shape together.
# CartPole shape is (4,), so np.prod((4,)) = 4
# For an image input with shape (84, 84, 3), np.prod = 21168
```

---

## Summary: What Each Import Provides

```
┌─────────────────────────────────────────────────────────────────┐
│ Import                            │ What We Use From It        │
├─────────────────────────────────────────────────────────────────┤
│ torch.nn as nn                    │ nn.Module      (base class)│
│                                   │ nn.Sequential  (chain)     │
│                                   │ nn.Linear      (layer)     │
│                                   │ nn.Tanh        (activation)│
│                                   │ nn.init.*      (init)      │
├─────────────────────────────────────────────────────────────────┤
│ torch.distributions.categorical   │ Categorical    (action     │
│   as Categorical                  │   distribution for         │
│                                   │   sampling & log_prob)     │
├─────────────────────────────────────────────────────────────────┤
│ numpy as np                       │ np.sqrt(2)     (init std)  │
│                                   │ np.prod()      (obs shape) │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 1: Neural Networks (The Building Block)

## What is a Neural Network?

A neural network is a function that takes numbers in and produces numbers out.
It **learns** the right transformation by adjusting internal numbers called **weights**.

Think of it as a pipeline of simple math operations:

```
    INPUT (numbers)
          │
          ▼
┌──────────────────────┐
│ Multiply by weights  │    y = W × x + b
│ Add bias             │
│ (this is "Linear")   │    W = weights (learned)
└──────────┬───────────┘    b = bias (learned)
           │
           ▼
┌──────────────────────┐
│ Activation function  │    Squash the result into a range
│ (e.g. Tanh)          │    Tanh → output between -1 and +1
└──────────┬───────────┘
           │
           ▼

     ... repeat ...         Stack more layers = "deep" learning

           │
           ▼
       OUTPUT (numbers)

```

### A Concrete Example

CartPole gives us 4 numbers as observation:

```
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
     0.02          0.15          -0.03            -0.12
```

We want the network to output: "should I push left or right?"

```
[0.02, 0.15, -0.03, -0.12]
           │
           ▼
    Linear(4 → 64)         4 inputs → 64 outputs
    (multiply by a 64×4 matrix of weights, add 64 biases)
           │
           ▼
    [0.83, -0.42, 1.21, ..., -0.67]    ← 64 numbers
           │
           ▼
    Tanh()
    [0.68, -0.40, 0.84, ..., -0.58]    ← 64 numbers, now between -1 and +1
           │
           ▼
    Linear(64 → 64)
           │
           ▼
    Tanh()
           │
           ▼
    Linear(64 → 2)         64 inputs → 2 outputs (left, right)
           │
           ▼
    [1.5, -0.3]            ← "logits": raw scores for each action
                              higher score = network prefers this action
                              1.5 (left) > -0.3 (right) → probably push left
```

### What is "Learning"?

The weights start random. The network gives bad answers.
We measure HOW bad (the "loss"), then nudge the weights slightly
to make the answer less bad. Repeat millions of times.

```
                    ┌─────────────────────────────┐
                    │                             │
                    ▼                             │
  Input ──► Network(weights) ──► Output           │
                                   │              │
                                   ▼              │
                              Compare with        │
                              desired result      │
                                   │              │
                                   ▼              │
                                 Loss             │
                              (how bad?)          │
                                   │              │
                                   ▼              │
                              Adjust weights ─────┘
                              (gradient descent)
```

### What is Tanh?

Tanh is an "activation function" — it squashes any number into the range [-1, +1]:

```
Input:     -100    -2     -1      0      1      2     100
Tanh:      -1.0  -0.96  -0.76   0.0   0.76   0.96   1.0

Without activation functions, stacking layers would be pointless —
multiple linear operations collapse into a single linear operation.
Tanh adds the non-linearity that lets networks learn complex patterns.
```

---

# Part 2: Reinforcement Learning (The Problem)

## The Setup

Unlike regular ML where you have labeled data, in Reinforcement Learning (RL)
an **agent** learns by **interacting** with an **environment**:

```
┌───────────┐    action     ┌─────────────┐
│           │──────────────►│             │
│   AGENT   │               │ ENVIRONMENT │
│  (brain)  │◄──────────────│  (world)    │
│           │  observation  │             │
│           │  + reward     │             │
└───────────┘               └─────────────┘
```

- **Agent**: the decision-maker (our neural network)
- **Environment**: the world (CartPole game)
- **Observation**: what the agent sees (pole angle, cart position, etc.)
- **Action**: what the agent does (push left or push right)
- **Reward**: feedback signal (+1 for each step the pole stays up)

## The Goal

Maximize total reward over time.

```
Step 1: pole is up     → reward = +1
Step 2: pole is up     → reward = +1
Step 3: pole is up     → reward = +1
...
Step 47: pole falls    → episode ends

Total reward = 47 (bad — max is 500)

After training:
Step 1-500: pole stays up → total reward = 500 (perfect!)
```

## Key Concepts

### Policy (π)

The **policy** is the agent's strategy — a mapping from observation to action.

```
Policy: "Given what I see, what should I do?"

observation [0.02, 0.15, -0.03, -0.12]
                    │
                    ▼
              Policy (π)
                    │
                    ▼
            action: push LEFT
```

A neural network IS the policy. It takes observations and outputs actions.

### Value (V)

The **value** of a state is "how much total future reward do I expect from here?"

```
State: pole is perfectly balanced
Value: ~450   (we expect ~450 more reward from here)

State: pole is about to fall
Value: ~3     (we expect only ~3 more reward)
```

Value helps the agent judge: "Am I in a good situation or a bad one?"

### Advantage (A)

**Advantage** = "Was this action better or worse than average?"

```

Advantage = (actual reward I got) - (reward I expected)

If I expected value 100 from this state, and I got 120:
    Advantage = +20  → "That action was BETTER than average, do more of it"

If I expected value 100 from this state, and I got 80:
    Advantage = -20  → "That action was WORSE than average, do less of it"

```

---

# Part 3: Actor-Critic (The Architecture)

## Why Two Networks?

We need two things:

1. A way to **choose actions** → the Actor (policy)
2. A way to **evaluate states** → the Critic (value)

```
              observation
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   ┌─────────┐        ┌─────────┐
   │  ACTOR  │        │ CRITIC  │
   │         │        │         │
   │ "What   │        │ "How    │
   │ should  │        │ good is │
   │ I do?"  │        │ this    │
   │         │        │ state?" │
   └────┬────┘        └────┬────┘
        │                  │
        ▼                  ▼
     ACTION             VALUE
   (left/right)     (expected reward)
```

### The Actor (Policy Network)

```
Input:  observation (4 numbers for CartPole)
Output: probability of each action

Example:
  [0.02, 0.15, -0.03, -0.12]  →  [0.85, 0.15]
                                    ↑      ↑
                                  LEFT   RIGHT
                                  85%     15%

  Then we SAMPLE from this distribution:
  85% chance we pick LEFT, 15% chance we pick RIGHT

  Why sample instead of always picking the best?
  → EXPLORATION. Sometimes trying "bad" actions leads
    to discovering they're actually good.
```

### The Critic (Value Network)

```
Input:  observation (4 numbers for CartPole)
Output: single number (estimated future reward)

Example:
  [0.02, 0.15, -0.03, -0.12]  →  347.5
                                    ↑
                                  "I expect about 347 more
                                   reward from this state"
```

### How They Help Each Other

```
1. Actor picks action → gets reward from environment
2. Critic evaluates: "Was that good or bad?"

   Advantage = actual_reward - critic_predicted_value

3. If advantage > 0:
     Tell Actor: "Good job! Do this action MORE often"
   If advantage < 0:
     Tell Actor: "Bad move! Do this action LESS often"

4. Also update Critic to predict better next time

This is more stable than using only an Actor because:
- Actor alone has HIGH VARIANCE (rewards are noisy)
- Critic provides a BASELINE that reduces noise
- Together = faster, more stable learning
```

---

# Part 4: PPO (Proximal Policy Optimization)

## The Problem PPO Solves

Updating a policy is dangerous. If you change it too much in one step,
performance can collapse and never recover:

```
Performance
    │
    │   ╱╲        ← one bad update
    │  ╱  ╲
    │ ╱    ╲
    │╱      ╲________ ← collapsed, can't recover
    │
    └────────────────── Training steps

PPO prevents this by CLIPPING the update — never changing too much at once:

Performance
    │          ╱──────── ← steady improvement
    │        ╱
    │      ╱
    │    ╱
    │  ╱
    │╱
    └────────────────── Training steps
```

## How PPO Works (Step by Step)

### Step 1: Collect Experience (Rollout)

Run the current policy in the environment and record everything:

```
┌─────┬──────────────────┬────────┬────────┬───────────┐
│Step │ Observation      │ Action │ Reward │ log_prob  │
├─────┼──────────────────┼────────┼────────┼───────────┤
│  1  │ [0.02, 0.15,...] │ LEFT   │  +1    │ -0.16     │
│  2  │ [0.05, 0.11,...] │ RIGHT  │  +1    │ -0.69     │
│  3  │ [0.03, 0.08,...] │ LEFT   │  +1    │ -0.22     │
│ ... │ ...              │ ...    │  ...   │ ...       │
│ 128 │ [0.11,-0.03,...] │ RIGHT  │  +1    │ -0.51     │
└─────┴──────────────────┴────────┴────────┴───────────┘

log_prob = how likely was this action under the CURRENT policy
           (we save this as "old" log_prob for later)
```

### Step 2: Compute Advantages

For each step, calculate "was this action better or worse than expected?"

```
For each step t:
    advantage_t = (reward_t + future_rewards) - critic_value_t
                   \_______________________/
                    what actually happened
                                             \____________/
                                             what critic predicted
```

### Step 3: Update the Policy (The PPO Trick)

Now we want to update the actor to do MORE of the good actions
and LESS of the bad actions. But not too aggressively.

```
ratio = new_prob(action) / old_prob(action)

If ratio = 1.0  → policy hasn't changed for this action
If ratio = 1.5  → policy now thinks this action is 50% MORE likely
If ratio = 0.5  → policy now thinks this action is 50% LESS likely
```

PPO clips this ratio to stay within [1-ε, 1+ε] (ε is usually 0.2):

```
    Without clipping:              With PPO clipping (ε=0.2):

    ratio can go anywhere:         ratio is clamped to [0.8, 1.2]:

    ──────────────────────         ──────────────────────
    0    0.5   1.0   1.5          0.8  0.9  1.0  1.1  1.2
              ↑                              ↑
         no change                      no change

    Huge updates possible!         Small, safe updates only!
```

The actual loss formula:

```
L = min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)

This means:
- If advantage > 0 (good action):
    increase probability, but cap at ratio = 1.2

- If advantage < 0 (bad action):
    decrease probability, but cap at ratio = 0.8

→ Policy changes slowly and safely
```

### Step 4: Repeat

```
┌──────────────────┐
│ Collect rollout  │  ◄────────────────────────┐
│ (run policy)     │                           │
└────────┬─────────┘                           │
         ▼                                     │
┌──────────────────┐                           │
│ Compute          │                           │
│ advantages       │                           │
└────────┬─────────┘                           │
         ▼                                     │
┌──────────────────┐                           │
│ PPO update       │ × 4 epochs                │
│ (clip + optimize)│ (reuse same rollout data) │
└────────┬─────────┘                           │
         ▼                                     │
   Better policy ──────────────────────────────┘
```

---

# Part 5: The Code (`src/agent.py`) Explained

Now you have all the background. Let's read the actual code.

## `layer_init()` — Preparing the Weights

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
```

Before training starts, weights need sensible starting values.
Random noise works, but **orthogonal initialization** works better —
it prevents signals from growing or shrinking as they pass through layers.

```
                    std value      Why?
                    ─────────      ──────────────────────────────
Hidden layers:      √2 ≈ 1.41     Compensates for Tanh squashing.
                                   Keeps signal strength stable.

Actor output:       0.01           Tiny weights → outputs near zero →
                                   all actions equally likely at start →
                                   agent EXPLORES before committing.

Critic output:      1.0            Normal scale. Value predictions
                                   can be any magnitude.
```

## The `Agent` Class — The Brain

```python
class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
```

`nn.Module` is PyTorch's base class for neural networks.
Inheriting from it gives us automatic weight tracking, GPU support, etc.

### The Critic Network

```python
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),     # 4 → 64
            nn.Tanh(),                               # squash to [-1, 1]
            layer_init(nn.Linear(64, 64)),           # 64 → 64
            nn.Tanh(),                               # squash again
            layer_init(nn.Linear(64, 1), std=1.0),   # 64 → 1 (single value)
        )
```

```
Observation          Hidden          Hidden           Value
[4 numbers] ──► [64 numbers] ──► [64 numbers] ──► [1 number]
             Linear+Tanh      Linear+Tanh        Linear
                                              (std=1.0)

Output: "I think this state is worth 347.5 future reward"
```

### The Actor Network

```python
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),              # 4 → 64
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),                   # 64 → 64
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),  # 64 → 2
        )
```

```
Observation          Hidden          Hidden          Logits
[4 numbers] ──► [64 numbers] ──► [64 numbers] ──► [2 numbers]
             Linear+Tanh      Linear+Tanh        Linear
                                              (std=0.01)

Output: [1.5, -0.3] → "I slightly prefer LEFT over RIGHT"

These logits become probabilities via softmax:
  softmax([1.5, -0.3]) = [0.86, 0.14]
  (This happens inside Categorical distribution automatically)
```

### `get_value()` — Ask the Critic

```python
    def get_value(self, x):
        return self.critic(x)
```

Just runs the observation through the critic network.
Used when we need ONLY the value (e.g., for bootstrapping at end of rollout).

### `get_action_and_value()` — The Main Method

```python
    def get_action_and_value(self, x, action=None):
        action_logits = self.actor(x)
        distribution = Categorical(logits=action_logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(x)
```

Step by step:

```
1. Run actor:        logits = actor(observation)      → [1.5, -0.3]
2. Make distribution: Categorical(logits=[1.5, -0.3]) → P(left)=86%, P(right)=14%
3. Sample action:    action = distribution.sample()   → LEFT (probably)
4. Compute log_prob: log(0.86)                        → -0.15
5. Compute entropy:  -(0.86·log(0.86) + 0.14·log(0.14)) → 0.43
6. Run critic:       value = critic(observation)      → 347.5

Returns: (LEFT, -0.15, 0.43, 347.5)
```

**Why `action` parameter?**

```
During ROLLOUT:     action=None  → sample a new action (explore!)
During PPO UPDATE:  action=LEFT  → don't sample, just compute log_prob
                                    of the action we already took
                                    (needed for the ratio calculation)
```

**Why entropy?**

```
Entropy = measure of randomness in the distribution

High entropy: [0.5, 0.5]     → very uncertain, exploring a lot
Low entropy:  [0.99, 0.01]   → very confident, barely exploring

PPO adds entropy as a BONUS to the loss:
  loss = policy_loss + value_loss - entropy_bonus
                                    ↑
                                    encourages exploration
                                    prevents premature convergence
```

---

## Complete Picture

```
                        ┌─────────────────────┐
                        │    ENVIRONMENT      │
                        │    (CartPole)       │
                        └──┬──────────────┬───┘
                   obs     │              │  reward
                   ┌───────┘              └────────┐
                   │                               │
                   ▼                               ▼
            ┌─────────────┐                  ┌───────────┐
      ┌────►│    AGENT    │                  │ Advantage │
      │     │             │                  │ Compute   │
      │     │ ┌─────────┐ │   value          │           │
      │     │ │ CRITIC  ├─┼──────────────────► A = R - V │
      │     │ └─────────┘ │                  └─────┬─────┘
      │     │             │                        │
      │     │ ┌─────────┐ │   action               │
      │     │ │  ACTOR  ├─┼────────► env.step()    │
      │     │ └─────────┘ │                        │
      │     └─────────────┘                        │
      │                                            │
      │     ┌──────────────────────────────────────┘
      │     │  PPO Update:
      │     │  ratio = new_prob / old_prob
      │     │  loss = -min(ratio·A, clip(ratio)·A)
      │     │  optimizer.step() → adjust weights
      │     │
      └─────┘  (repeat with improved policy)
```

---

## Note: Current Code is Incomplete

The `get_action_and_value` method in the file is unfinished.
It creates the distribution but doesn't return anything.
The complete version should be:

```python
def get_action_and_value(self, x, action=None):
    action_logits = self.actor(x)
    distribution = Categorical(logits=action_logits)
    if action is None:
        action = distribution.sample()
    return action, distribution.log_prob(action), distribution.entropy(), self.critic(x)
```

