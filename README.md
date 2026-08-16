# A-EVIS

**Amortized Environment-Verified Inductive Search**

A-EVIS is a research prototype for a candidate **non-RL interaction-learning framework**. Instead of assigning temporal credit to individual actions with value functions or policy gradients, A-EVIS treats **complete executable behaviors/programs as hypotheses**, verifies them in an environment, and reuses previously verified programs across tasks.

> **Status:** early research prototype. The current evidence is promising but limited. A-EVIS should not yet be described as a proven replacement for reinforcement learning.

## Core idea

Conventional RL commonly learns through a temporal-credit mechanism such as

```text
trajectory -> reward / return -> value or advantage -> parameter update
```

A-EVIS instead uses

```text
complete program hypothesis
        -> environment verification
        -> retain / reject
        -> verified-program library
        -> retrieve + adapt on future tasks
```

The current prototype deliberately avoids:

- Q-functions and V-functions
- Bellman or temporal-difference updates
- policy gradients, advantages, critics, PPO clipping
- behavioral distillation or teacher demonstrations

The environment is used to evaluate **whole candidate programs**.

## Why amortization matters

The original EVIS prototype could solve a new task by searching for a verified program, but it had to repeat much of that search on every new task.

A-EVIS adds a verified cross-task memory:

```text
(task fingerprint, verified program, episode outcome)
```

For a new task, A-EVIS retrieves similar verified programs, forms a nonparametric program proposal, verifies a few complete candidates, and runs local whole-program search only when needed.

## Current sandbox benchmark: RobotReach

RobotReach is a self-contained Isaac-style continuous-control benchmark with randomized dynamics, continuous torques, sensor noise, and unseen target configurations.

On a 120-task stream with a ~60k interaction budget (3 seeds):

| Method | Mean interactions | Success rate |
|---|---:|---:|
| **A-EVIS** | **52.4k** | **60.6% ± 4.6%** |
| Cold EVIS (no cross-task reuse) | 59.3k | 46.7% ± 2.9% |
| TD3 | 60.0k | 48.6% ± 0.5% |
| PPO | 60.0k | 28.1% ± 2.9% |

The strictest comparison is **A-EVIS vs Cold EVIS**, because they operate on the same sequential-task protocol. Across the three seeds, amortization improved success by +15.0, +17.5, and +9.2 percentage points while using fewer interactions.

## Standard Atari / ALE validation — pending local results

The repository now includes a standard Atari validation harness for local reproduction. **No standard ALE score is claimed yet.** The first protocol uses `ALE/*-v5`, sticky actions, action repeat 4, 84x84 grayscale frame stacks, and exact ALE clone/restore for A-EVIS counterfactual whole-program evaluation.

Initial local suite:

```text
ALE/Pong-v5
ALE/Breakout-v5
ALE/Seaquest-v5
ALE/BeamRider-v5
```

Install the optional Atari dependencies:

```bash
pip install -e '.[atari]'
```

Then run A-EVIS:

```bash
python experiments/atari/run_aevis.py \
  --game ALE/Pong-v5 \
  --seed 0 \
  --frame-budget 1000000 \
  --out results/atari/aevis_pong_seed0.csv
```

Reference PPO/DQN runners are also included:

```bash
python experiments/atari/run_sb3.py \
  --algo ppo --game ALE/Pong-v5 --seed 0 \
  --agent-steps 250000 \
  --out results/atari/ppo_pong_seed0.csv

python experiments/atari/run_sb3.py \
  --algo dqn --game ALE/Pong-v5 --seed 0 \
  --agent-steps 250000 \
  --out results/atari/dqn_pong_seed0.csv
```

A-EVIS counts **every candidate branch rollout** in the raw emulator-frame budget. Missing Atari runs remain explicitly marked as pending rather than being filled with sandbox results.

See [`docs/ATARI.md`](docs/ATARI.md) for the full protocol and [`results/atari/README.md`](results/atari/README.md) for the pending result table.

## Repository layout

```text
src/aevis/              generic verified-program utilities
benchmarks/             RobotReach + cloneable standard Atari adapter
experiments/             runnable prototype comparisons
experiments/atari/       A-EVIS and PPO/DQN Atari runners
results/                 checked-in summary results
results/atari/           locally reproduced Atari results (currently pending)
docs/                    method, results, and Atari protocol notes
tests/                   lightweight smoke tests
```

## Quick start

Core utilities only require NumPy:

```bash
pip install -e .
```

RL baselines additionally require PyTorch:

```bash
pip install -e '.[rl]'
```

Atari validation:

```bash
pip install -e '.[atari]'
```

Atari ROM availability/licensing is the user's responsibility; this repository does not download or redistribute ROMs.

## What is general, and what is not yet proven

The **learning rule** is intended to be domain-agnostic:

1. represent a complete executable behavior/program;
2. evaluate the whole program in the environment;
3. retain environment-verified hypotheses;
4. reuse verified programs across related tasks;
5. locally refine complete programs when retrieval is insufficient.

The action-program representation currently differs by domain. In the first Atari harness, A-EVIS searches a low-dimensional latent program decoded by a fixed random hypernetwork over generic pooled pixel features. A major open question is whether a single sufficiently expressive program representation can scale across standard games, high-dimensional robotics, and verifiable LLM agents without introducing domain-specific learning machinery.

## Claims and reproducibility

The RobotReach numbers in this repository come from sandbox prototype experiments. They are **not peer reviewed**, and several comparisons use different deployment/training protocols. Standard Atari results are deliberately left pending until reproduced locally. The repository therefore treats A-EVIS as a **candidate learning framework**, not an established new paradigm.

See `docs/METHOD.md`, `docs/RESULTS.md`, and `docs/ATARI.md` for details.
