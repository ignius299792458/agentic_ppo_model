# Glossary — Every Term You'll Encounter in This Project

---

## Data & Math Fundamentals

### Tensor

A multi-dimensional array of numbers. Generalization of scalars, vectors, and matrices.

```
Scalar:  42                         → 0 dimensions
Vector:  [1.0, 2.0, 3.0]            → 1 dimension   (shape: 3)
Matrix:  [[1, 2], [3, 4]]           → 2 dimensions  (shape: 2×2)
Tensor:  a cube of numbers          → 3+ dimensions (shape: 200×8×4)
```

PyTorch tensors are like NumPy arrays but can run on GPU and track gradients.

### Shape

The size of each dimension of a tensor. Tells you how the numbers are organized.

```
torch.zeros(200, 8, 4)  → shape is (200, 8, 4)
                            200 timesteps, 8 envs, 4 obs values
```

### Batch

A group of data samples processed together. Instead of training on one example
at a time, we process many at once for efficiency.

```
batch_size = num_steps × num_envs = 200 × 8 = 1600 transitions
```

### Minibatch

A smaller chunk of the full batch. We split the batch into minibatches
and update the network on each one separately.

```
batch (1600) ÷ num_minibatches (4) = minibatch_size (400)
```

### Epoch

One complete pass through all the data. In PPO, we do multiple epochs
(default 4) over the same rollout data before collecting new data.

### Flatten

Reshaping a multi-dimensional tensor into a single dimension.

```
(200, 8)  →  flatten  →  (1600,)
Each transition becomes an independent training example.
```

### Gradient

The direction and magnitude to nudge each weight to reduce the loss.
Computed automatically by PyTorch via backpropagation.

### Gradient Descent

The algorithm that updates weights by moving them in the opposite
direction of the gradient (downhill on the loss surface).

```
weight_new = weight_old - learning_rate × gradient
```

### Learning Rate (LR)

How big each weight update step is. Too large = unstable. Too small = slow.
Default: 2.5e-4 (0.00025).

### LR Annealing

Gradually reducing the learning rate over training. Starts large (fast learning),
ends at zero (fine-tuning). Linear annealing:

```
lr_t = lr_initial × (1 - t / T)
```

### Loss

A single number measuring "how wrong" the network is. Training minimizes this.
Lower loss = better. PPO has three loss components: policy, value, and entropy.

### Backpropagation

Algorithm that computes gradients of the loss with respect to every weight
by applying the chain rule backwards through the network.

### Optimizer (Adam)

Algorithm that applies gradients to update weights. Adam adapts the learning
rate per-weight based on gradient history, handling different scales automatically.

---

## Neural Network Terms

### Layer

A single computation step in a neural network. Our networks use Linear layers.

### Linear Layer

Multiplies input by a weight matrix and adds a bias:

```
output = W × input + b

nn.Linear(4, 64)  →  4 inputs, 64 outputs
                      W is a 64×4 matrix (256 weights)
                      b is a 64-vector (64 biases)
                      Total: 320 learnable parameters
```

### Activation Function

A non-linear function applied after a linear layer. Without it,
stacking linear layers collapses into a single linear operation.

### Tanh

Activation function that squashes any number to the range [-1, +1].

```
tanh(-100) = -1.0,  tanh(0) = 0.0,  tanh(100) = 1.0
```

### Weights

The learnable numbers inside a layer. Adjusted during training to
minimize the loss. A network with 2 hidden layers of size 64 has ~4,610 weights.

### Bias

A learnable constant added after the matrix multiplication. Lets the
network shift its output up or down.

### Parameters

All learnable numbers in a network — all weights and biases combined.

### nn.Module

PyTorch's base class for neural networks. Tracks all parameters,
handles moving to GPU, saving/loading, etc.

### nn.Sequential

A container that chains layers in order. Data flows through each layer
in sequence: layer1 → activation → layer2 → activation → layer3.

### Orthogonal Initialization

Setting initial weights to an orthogonal matrix (rows are perpendicular).
Preserves signal magnitude across layers, preventing vanishing/exploding gradients.

### Gradient Clipping

Capping the total gradient magnitude to prevent huge updates:

```
if ||gradient|| > max_norm:
    gradient = gradient × (max_norm / ||gradient||)
```

---

## Reinforcement Learning Terms

### Agent

The decision-maker. In our code, it's the `Agent` class — a neural network
that observes the environment and chooses actions.

### Environment (Env)

The world the agent interacts with. CartPole is a pole balanced on a cart.
The agent pushes left or right to keep it upright.

### Observation (obs / state)

What the agent can see. CartPole provides 4 numbers:

```
[cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

### Action

What the agent does. CartPole has 2 discrete actions: push LEFT (0) or RIGHT (1).

### Reward

Feedback signal from the environment. CartPole gives +1 for every step
the pole stays upright. Episode ends when it falls.

### Episode

One complete game from start to finish. Starts with env.reset(),
ends when terminated or truncated.

### Terminated

Episode ended because of a game rule (pole fell, goal reached).

### Truncated

Episode ended because of a time limit (max 500 steps in CartPole),
not because the agent failed.

### Done

Episode ended for either reason: done = terminated OR truncated.

### Step

One interaction cycle: agent picks action → environment returns
new observation + reward + done.

### Timestep

Same as step. `global_step` counts total steps across all environments.

### Episodic Return

Total reward accumulated in one episode. For CartPole, this is the number
of steps the pole stayed up (since each step gives +1).

### Rollout

The phase where the agent plays the game and we record everything.
No learning happens during rollout — we're just collecting data.

```
200 steps × 8 envs = 1600 transitions recorded per rollout
```

### Transition

A single (obs, action, reward, next_obs, done) tuple. One step of experience.

### Vectorized Environment

Multiple copies of the environment running simultaneously.
SyncVectorEnv runs N envs in one process, returning batched results.

```
8 envs → one env.step() returns 8 obs, 8 rewards, 8 dones
```

### Render Mode

How the environment displays itself:

```
None:        no rendering (fastest)
"human":     live window (pygame) — watch the agent play
"rgb_array": returns pixel frames — for video recording
```

---

## Policy & Value Terms

### Policy (pi)

The agent's strategy: a mapping from observation to action probabilities.

```
pi(LEFT | obs) = 0.86    "86% chance I choose LEFT"
pi(RIGHT | obs) = 0.14   "14% chance I choose RIGHT"
```

Our actor network IS the policy.

### Actor

The neural network that outputs action probabilities (the policy).

### Critic

The neural network that estimates how good a state is (the value function).

### Actor-Critic

Architecture with two networks: actor (what to do) + critic (how good is this).
They help each other — critic provides a baseline that reduces noise in the
actor's learning signal.

### Value (V)

The expected total future reward from a state, following the current policy:

```
V(s) = E[r_0 + gamma*r_1 + gamma^2*r_2 + ... | starting from s]

"If I'm in this state and keep following my policy, how much total
 reward do I expect to get?"
```

### State Value vs Action Value

- V(s): value of being in state s (our critic outputs this)
- Q(s,a): value of taking action a in state s, then following policy

### Advantage (A)

How much better an action was compared to the average:

```
A(s,a) = Q(s,a) - V(s)

A > 0: "this action was BETTER than average, do it more"
A < 0: "this action was WORSE than average, do it less"
A = 0: "this action was exactly average"
```

### Return (G)

The actual total discounted reward from a timestep onward:

```
G_t = r_t + gamma × r_{t+1} + gamma^2 × r_{t+2} + ...
```

### Discount Factor (gamma)

How much future rewards are worth compared to immediate rewards.
gamma = 0.99 means a reward 100 steps away is worth 0.99^100 = 0.37 now.

```
gamma = 0:    only care about immediate reward (myopic)
gamma = 1:    future rewards worth as much as now (far-sighted)
gamma = 0.99: sweet spot — care about future but prefer sooner
```

### Bootstrapping

Estimating future reward using the critic's prediction instead of
waiting for actual rewards. V(s_{t+1}) "bootstraps" the return estimate.

---

## PPO-Specific Terms

### PPO (Proximal Policy Optimization)

A policy gradient algorithm that limits how much the policy changes per
update using a clipping mechanism. Stable, simple, and effective.

### On-Policy

Learning only from data collected by the current policy. After each PPO
update, old data is discarded and new data is collected. (Contrast with
off-policy methods like DQN that reuse old data.)

### Probability Ratio

How much more/less likely an action is under the new policy vs the old:

```
r = pi_new(a|s) / pi_old(a|s)

r = 1.0: policy unchanged for this action
r = 1.5: action is now 50% more likely
r = 0.5: action is now 50% less likely
```

### Clipping (PPO Clip)

Capping the probability ratio to [1-eps, 1+eps] (default eps=0.2,
so [0.8, 1.2]). Prevents the policy from changing too drastically in
one update, which could collapse performance.

### Clip Coefficient (epsilon / clip_coef)

The clipping range. Default 0.2 means ratios are capped to [0.8, 1.2].
Smaller = more conservative updates. Larger = bigger steps allowed.

### Surrogate Objective

The loss function PPO optimizes. "Surrogate" because it approximates
the true policy improvement using the probability ratio instead of
directly optimizing expected reward.

### KL Divergence (approx_kl)

Measures how different the new policy is from the old policy.
Large KL = policy changed a lot. PPO clip implicitly limits KL.

### Clip Fraction (clipfrac)

Percentage of transitions where the ratio was clipped. Healthy range
is 0.1-0.3. If too high, the policy is trying to change too fast.

---

## GAE Terms

### TD Error (delta)

One-step advantage estimate. Uses one real reward plus the critic's
estimate of the next state:

```
delta_t = r_t + gamma × V(s_{t+1}) - V(s_t)
          ├─────────────────────┤   ├──────┤
          what actually happened   what we expected
```

### GAE (Generalized Advantage Estimation)

A weighted average of multi-step TD errors that balances bias vs variance:

```
A_t = delta_t + (gamma × lambda) × delta_{t+1} + (gamma × lambda)^2 × delta_{t+2} + ...
```

### Lambda (gae_lambda)

Controls the GAE bias-variance tradeoff:

```
lambda = 0:    A = delta_t (one-step TD, low variance, high bias)
lambda = 1:    A = G_t - V(s_t) (Monte Carlo, high variance, low bias)
lambda = 0.95: balanced — the standard choice
```

### Bias

Systematic error from using estimates (like V) instead of true values.
The critic might be wrong, which biases the advantage estimate.

### Variance

Random noise in the estimate. Using many future rewards (Monte Carlo)
is unbiased but noisy because rewards are stochastic.

---

## Loss Components

### Policy Loss (pg_loss)

The PPO clipped surrogate objective. Measures how to improve the policy:

```
L = min(ratio × advantage, clip(ratio, 1-eps, 1+eps) × advantage)
```

### Value Loss (v_loss)

How wrong the critic's predictions are:

```
L_V = 0.5 × (V(s) - R_t)^2
```

### Entropy Loss / Entropy Bonus

Entropy measures randomness of the policy distribution:

```
H(pi) = -sum pi(a) × log pi(a)
```

Added as a bonus (subtracted from loss) to encourage exploration.
Without it, the policy might converge to always picking one action.

### Entropy Coefficient (ent_coef)

Weight of the entropy bonus. Default 0.01. Higher = more exploration.

### Value Coefficient (vf_coef)

Weight of the value loss. Default 0.5. Balances policy vs value learning.

### Total Loss

```
L = policy_loss + vf_coef × value_loss - ent_coef × entropy
```

---

## Probability & Distribution Terms

### Logits

Raw, unnormalized scores output by the actor network. Can be any number.
Converted to probabilities by softmax.

```
logits: [1.5, -0.3]  →  softmax  →  probabilities: [0.86, 0.14]
```

### Softmax

Converts logits to a probability distribution (positive, sums to 1):

```
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

### Categorical Distribution

A probability distribution over a finite set of discrete choices.
Created from logits. Supports sampling, log_prob, and entropy.

### Log Probability (log_prob)

The natural log of the probability of an action:

```
log pi(LEFT|s) = log(0.86) = -0.15
```

Used instead of raw probabilities because:

- Multiplications become additions (numerically stable)
- PPO ratio = exp(log_new - log_old) instead of new/old

### Entropy (of a distribution)

Measure of uncertainty/randomness:

```
H = -sum pi(a) × log pi(a)

[0.5, 0.5]:    H = 0.69  (maximum uncertainty, 2 actions)
[0.86, 0.14]:  H = 0.41  (somewhat certain)
[0.99, 0.01]:  H = 0.06  (very certain)
```

### Sampling

Randomly picking an action according to the probability distribution.
With [0.86, 0.14], we pick LEFT 86% of the time, RIGHT 14%.

---

## Training & Logging Terms

### Global Step

Running count of total environment steps across all envs and all updates.
Used as the x-axis for TensorBoard plots.

### Update

One cycle of: rollout → advantages → PPO update. The number of updates is:

```
num_updates = total_timesteps / batch_size
            = 25000 / 1600
            = 15 updates
```

### SPS (Steps Per Second)

Throughput metric. How many environment steps are processed per second.

### Explained Variance

How well the critic predicts actual returns. Ranges from -inf to 1.0:

```
1.0:  perfect predictions
0.0:  predictions are no better than guessing the mean
<0:   predictions are worse than guessing the mean
```

### TensorBoard

Visualization tool for training metrics. Run `tensorboard --logdir runs`
to view charts of returns, losses, etc. in your browser.

### Weights & Biases (W&B / wandb)

Cloud-based experiment tracking platform. Alternative to TensorBoard
with collaboration features. Enabled with `--track`.

---

## Environment-Specific Terms

### CartPole-v1

A classic control problem. A pole is attached to a cart on a rail.
The agent pushes left or right to keep the pole balanced.

```
Observation: [cart_pos, cart_vel, pole_angle, pole_angular_vel]  (4 numbers)
Actions:     0 (push left), 1 (push right)
Reward:      +1 per timestep the pole stays up
Termination: pole angle > 12° or cart leaves bounds
Max steps:   500 (truncation)
Solved:      average return >= 475 over 100 episodes
```

### RecordEpisodeStatistics

Gymnasium wrapper that tracks episode return (total reward) and length
(number of steps). Stores results in `info["episode"]`.

### RecordVideo

Gymnasium wrapper that captures frames and saves them as video files.
Requires `render_mode="rgb_array"` and `moviepy`.

### Seed

A number that initializes a random number generator to a known state.
Same seed = same sequence of random numbers = reproducible experiments.