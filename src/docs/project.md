# PPO Implementation — Complete Project Guide

## How Everything Fits Together

```
python -m src.main
       │
       ▼
   ┌─────────┐     ┌──────────┐     ┌──────────┐
   │ args.py │────►│  env.py  │────►│ agent.py │
   │(config) │     │ (world)  │     │ (brain)  │
   └─────────┘     └──────────┘     └──────────┘
       │                │                 │
       ▼                ▼                 ▼
   ┌──────────────────────────────────────────────┐
   │                 main.py                      │
   │                                              │
   │  for each update:                            │
   │    ┌────────────┐                            │
   │    │ rollout.py │  collect experience        │
   │    └─────┬──────┘                            │
   │          ▼                                   │
   │    ┌──────────────┐                          │
   │    │advantage.py  │  compute GAE             │
   │    └─────┬────────┘                          │
   │          ▼                                   │
   │    ┌──────────┐                              │
   │    │  ppo.py  │  minibatch clipped update    │
   │    └─────┬────┘                              │
   │          ▼                                   │
   │    ┌────────────┐                            │
   │    │ logger.py  │  write to TensorBoard      │
   │    └────────────┘                            │
   └──────────────────────────────────────────────┘
```

---

## Phase 1: Rollout (`rollout.py`)

### What happens

The agent interacts with the environment for `num_steps` steps across
`num_envs` parallel environments, collecting a batch of experience.

```
For step = 0 to num_steps-1:
    action ~ pi(·|obs)              ← sample from policy
    obs', reward, done = env.step(action)
    store: obs, action, log_prob, reward, done, value
```

### The math: Policy sampling

The actor network outputs **logits** z for each action:

    z = Actor(obs)        e.g. z = [1.5, -0.3] for 2 actions

These become probabilities via **softmax**:

    pi(a|s) = exp(z_a) / sum_i exp(z_i)

    pi(LEFT|s)  = exp(1.5) / (exp(1.5) + exp(-0.3)) = 4.48 / 5.22 = 0.86
    pi(RIGHT|s) = exp(-0.3) / (exp(1.5) + exp(-0.3)) = 0.74 / 5.22 = 0.14

We **sample** from this Categorical distribution rather than always picking
the highest probability action, to balance exploration vs exploitation.

We also store the **log probability** of the chosen action:

    log pi(a|s) = log(0.86) = -0.15

This is used later in the PPO ratio calculation.

---

## Phase 2: Advantage Estimation (`advantage.py`)

### The problem

We need to answer: "Was this action better or worse than average?"

This is the **advantage**: A(s,a) = Q(s,a) - V(s)

where Q(s,a) is the actual value of taking action a in state s,
and V(s) is the average value of state s.

### Discounted returns

Future rewards are worth less than immediate rewards. The **discount factor**
gamma (0.99) controls this:

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

With gamma = 0.99:

- Reward now: worth 1.00
- Reward in 10 steps: worth 0.99^10 = 0.90
- Reward in 100 steps: worth 0.99^100 = 0.37

### Simple advantage

    A_t = G_t - V(s_t)

Problem: G_t uses actual future rewards which are noisy (high variance).

### GAE (Generalized Advantage Estimation)

GAE balances bias vs variance using parameter lambda (0.95).

**TD error** (one-step advantage):

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

This has low variance (only one real reward) but high bias (relies on
the critic's estimate V which may be wrong).

**GAE** is an exponentially-weighted sum of multi-step TD errors:

    A_t^GAE = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...

Equivalently, computed recursively (which is what our code does):

    A_T = delta_T                                          (last step)
    A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}

**Lambda controls the tradeoff:**

    lambda = 0:   A_t = delta_t (TD(0), low variance, high bias)
    lambda = 1:   A_t = G_t - V(s_t) (Monte Carlo, high variance, low bias)
    lambda = 0.95: sweet spot used in practice

**Returns** are then:

    R_t = A_t + V(s_t)

These returns serve as the target for the critic's value loss.

### Handling episode boundaries

When an episode ends (done=True), future rewards from the NEXT episode
shouldn't count. The `(1 - done)` term zeros out bootstrapping across
episode boundaries:

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)

---

## Phase 3: PPO Update (`ppo.py`)

### Flattening the batch

Rollout data has shape (num_steps, num_envs). We flatten to (batch_size,)
where batch_size = num_steps \* num_envs, treating each transition as
independent.

### Minibatching

Instead of using the entire batch at once:

    1. Shuffle all batch_size indices randomly
    2. Split into num_minibatches chunks of minibatch_size each
    3. Update on each chunk separately
    4. Repeat for update_epochs

This gives more gradient updates per rollout and reduces correlation
between samples in each gradient step.

```
Batch (1600 transitions):
    Shuffle → [947, 23, 1102, 456, ...]
    Split  → MB1[400] | MB2[400] | MB3[400] | MB4[400]
    Update on each MB, then reshuffle and repeat (4 epochs)
    = 4 epochs × 4 minibatches = 16 gradient steps per rollout
```

### The PPO clipped objective

**Core idea:** limit how much the policy can change per update.

**Probability ratio:**

    r_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
        = exp(log pi_new - log pi_old)

If r = 1: policy unchanged. If r > 1: action became more likely.

**Unclipped objective:**

    L = r_t * A_t

Problem: if A_t > 0 (good action), gradient pushes r_t → infinity.
One huge update can destroy the policy.

**PPO clip:**

    L^CLIP = min(r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t)

With eps = 0.2, the ratio is clamped to [0.8, 1.2]:

    If A > 0 (good action): r is capped at 1.2
        → probability increases at most 20%

    If A < 0 (bad action):  r is capped at 0.8
        → probability decreases at most 20%

The min() takes the more pessimistic (conservative) estimate,
preventing both over-optimization directions.

### Value loss

The critic should predict returns accurately:

    L_V = 0.5 * (V(s) - R_t)^2

**With value clipping** (optional, prevents large value updates):

    V_clipped = V_old + clip(V_new - V_old, -eps, +eps)
    L_V = 0.5 * max((V_new - R)^2, (V_clipped - R)^2)

### Entropy bonus

Entropy measures how "spread out" the policy is:

    H(pi) = -sum_a pi(a|s) * log pi(a|s)

High entropy = exploring many actions. Low entropy = committed to one action.

Adding entropy as a bonus encourages exploration:

    L_entropy = H(pi)   (we want to MAXIMIZE this, so we subtract it from loss)

### Total loss

    L = L^CLIP + c_v * L_V - c_e * H(pi)

Where c_v = 0.5 (value coefficient), c_e = 0.01 (entropy coefficient).

The minus sign on entropy makes it a bonus (optimizer minimizes loss,
so subtracting entropy = maximizing entropy).

### Gradient clipping

After computing gradients, we clip them by global norm:

    if ||grad|| > max_grad_norm:
        grad = grad * (max_grad_norm / ||grad||)

This prevents any single large gradient from destabilizing training.

---

## Phase 4: Logging (`logger.py`)

After each update, we log to TensorBoard:

| Metric               | What it means                                                |
| -------------------- | ------------------------------------------------------------ |
| `episodic_return`    | Total reward per episode (should increase)                   |
| `episodic_length`    | Steps per episode (should increase for CartPole)             |
| `learning_rate`      | Current LR (linearly anneals to 0)                           |
| `policy_loss`        | PPO clipped surrogate loss                                   |
| `value_loss`         | Critic MSE loss                                              |
| `entropy`            | Policy entropy (should slowly decrease)                      |
| `approx_kl`          | KL divergence between old and new policy (should stay small) |
| `clipfrac`           | Fraction of ratios that got clipped (0.1-0.3 is healthy)     |
| `explained_variance` | How well critic predicts returns (1.0 = perfect)             |
| `SPS`                | Steps per second (throughput)                                |

---

## The Neural Network (`agent.py`)

### Architecture

```
Actor:  obs(4) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(n_actions)
Critic: obs(4) → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1)
```

### Why Tanh?

Tanh squashes outputs to [-1, +1], adding non-linearity.
Without it, stacking linear layers collapses into a single linear function:

    W2 * (W1 * x) = (W2 * W1) * x = W_combined * x

Tanh between them lets the network learn non-linear decision boundaries.

### Orthogonal initialization

Standard random initialization can cause signals to grow or shrink
exponentially across layers. Orthogonal matrices preserve vector norms:

    ||Wx|| = ||x||    (for orthogonal W)

This keeps activations and gradients at a stable magnitude.

**Scale factors:**

    Hidden layers: std = sqrt(2)
        Tanh squashes the output variance. sqrt(2) pre-compensates so
        the variance after Tanh stays approximately 1.

    Actor output: std = 0.01
        Near-zero logits → softmax gives ~uniform distribution →
        agent explores all actions equally at the start.

    Critic output: std = 1.0
        No special scaling needed. Values can be any magnitude.

---

## LR Annealing

The learning rate decreases linearly from `lr` to 0 over training:

    lr_t = lr * (1 - t / T)

Where t is the current update and T is total updates.

Early in training, large LR allows rapid learning.
Late in training, small LR allows fine-tuning without oscillation.

---

## Vectorized Environments (`env.py`)

SyncVectorEnv runs N copies of the environment:

```
env_0: obs_0, reward_0, done_0 ──┐
env_1: obs_1, reward_1, done_1 ──┤
env_2: obs_2, reward_2, done_2 ──├──► batched obs (N, obs_dim)
...                               │      batched reward (N,)
env_7: obs_7, reward_7, done_7 ──┘      batched done (N,)
```

**Benefits:**

- More data per step (N transitions instead of 1)
- Different envs are at different points → less correlated data
- Environments auto-reset when done (vectorized env handles this)

---

## Full Data Flow (One Update)

```
                    ┌────────────────────────────────────┐
                    │         8 parallel envs             │
                    └──────────────┬─────────────────────┘
                                   │
                    ROLLOUT (200 steps × 8 envs = 1600 transitions)
                                   │
                    ┌──────────────▼─────────────────────┐
                    │ obs:      (200, 8, 4)              │
                    │ actions:  (200, 8)                  │
                    │ logprobs: (200, 8)                  │
                    │ rewards:  (200, 8)                  │
                    │ dones:    (200, 8)                  │
                    │ values:   (200, 8)                  │
                    └──────────────┬─────────────────────┘
                                   │
                    COMPUTE ADVANTAGES (GAE, backward pass)
                                   │
                    ┌──────────────▼─────────────────────┐
                    │ advantages: (200, 8)               │
                    │ returns:    (200, 8)               │
                    └──────────────┬─────────────────────┘
                                   │
                    FLATTEN to (1600,)
                                   │
                    PPO UPDATE (4 epochs × 4 minibatches = 16 gradient steps)
                                   │
                    ┌──────────────▼─────────────────────┐
                    │ Shuffle [0..1599]                   │
                    │ MB1 [400] → forward → loss → grad  │
                    │ MB2 [400] → forward → loss → grad  │
                    │ MB3 [400] → forward → loss → grad  │
                    │ MB4 [400] → forward → loss → grad  │
                    │ (repeat 4 epochs)                   │
                    └──────────────┬─────────────────────┘
                                   │
                    LOG metrics to TensorBoard
                                   │
                    ┌──────────────▼─────────────────────┐
                    │ Updated agent weights               │
                    │ → next rollout uses improved policy │
                    └────────────────────────────────────┘
```

---

## Key Equations Summary

| Concept                  | Equation                                     |
| ------------------------ | -------------------------------------------- |
| Softmax (logits → probs) | pi(a\|s) = exp(z_a) / sum_i exp(z_i)         |
| Log probability          | log pi(a\|s)                                 |
| Discounted return        | G*t = r_t + gamma \* G*{t+1}                 |
| TD error                 | delta*t = r_t + gamma \* V(s*{t+1}) - V(s_t) |
| GAE advantage            | A*t = delta_t + (gamma * lambda) * A*{t+1}   |
| Returns from GAE         | R_t = A_t + V(s_t)                           |
| Probability ratio        | r_t = pi_new(a\|s) / pi_old(a\|s)            |
| PPO clip objective       | L = min(r*A, clip(r, 1-eps, 1+eps)*A)        |
| Value loss               | L_V = 0.5 \* (V(s) - R_t)^2                  |
| Entropy                  | H = -sum pi(a) \* log pi(a)                  |
| Total loss               | L = L_clip + c_v*L_V - c_e*H                 |
| LR annealing             | lr_t = lr \* (1 - t/T)                       |
