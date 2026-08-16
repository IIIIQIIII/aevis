# Standard Atari / ALE validation

This repository contains a **local validation harness**, not checked-in Atari claims.
No standard-Atari result should be added to the README until it is reproduced on a
machine with a licensed/local ROM setup.

## Protocol

Default A-EVIS configuration:

- Gymnasium / ALE v5 environment IDs, e.g. `ALE/Pong-v5`
- RGB source observations
- `frameskip=1` in ALE and `action_repeat=4` in the repository adapter
- sticky actions: `repeat_action_probability=0.25`
- minimal action set: `full_action_space=False`
- max-pool last two frames, grayscale, 84x84, stack 4
- all A-EVIS branch rollouts count raw emulator frames

The ALE adapter snapshots both the emulator state and the local frame stack, so
counterfactual complete-program evaluations start from exactly the same agent
observation history.

## Why the Atari A-EVIS program is different from RobotReach

The learning rule is the same: **whole executable hypotheses are evaluated by the
environment and the verified-program archive is reused across tasks/episodes**.
Only the domain adapter changes.

For Atari, a candidate program is a 64-dimensional latent vector. A fixed random
hypernetwork decodes the latent into a linear controller over generic pooled pixel
features. The hypernetwork is never trained; A-EVIS searches the latent using complete
rollout return. This is deliberately modest so that the first experiment tests the
learning rule rather than hiding a large learned feature extractor inside A-EVIS.

## Installation

```bash
pip install -e '.[atari]'
```

You must make Atari ROMs available locally according to ALE/Gymnasium requirements.
The repository does not download or redistribute ROMs.

## Smoke test

The latent program/search tests do not need ALE or ROMs:

```bash
pytest tests/test_atari_program.py -q
```

For the environment adapter itself:

```bash
python - <<'PY'
from benchmarks.atari import AtariCloneEnv

env = AtariCloneEnv('ALE/Pong-v5', seed=0)
obs = env.reset()
snapshot = env.clone()
obs2, reward, done, info = env.step(0)
env.restore(snapshot)
env.close()
print(obs.shape)
PY
```

Expected stacked observation shape: `(4, 84, 84)`.

## A-EVIS

Start with Pong at a small budget:

```bash
python experiments/atari/run_aevis.py \
  --game ALE/Pong-v5 \
  --seed 0 \
  --frame-budget 1000000 \
  --out results/atari/aevis_pong_seed0.csv
```

Recommended initial suite:

```text
ALE/Pong-v5
ALE/Breakout-v5
ALE/Seaquest-v5
ALE/BeamRider-v5
```

Run at least 3 seeds before interpreting a result.

## Cold whole-program search ablation

This is the most important non-RL ablation. It uses **the exact same pixel
preprocessing, latent policy representation, and CEM whole-program search as
A-EVIS**, but it disables the cross-episode verified-program library.

```bash
python experiments/atari/run_cold.py \
  --game ALE/Pong-v5 \
  --seed 0 \
  --frame-budget 1000000 \
  --out results/atari/cold_pong_seed0.csv
```

If A-EVIS does not improve over this baseline, then the Atari result would support
black-box search but not the amortization mechanism that distinguishes A-EVIS.

## PPO and DQN reference baselines

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

The SB3 runner is a **reference baseline**, not a claim of best-known Atari
performance. For publication-grade comparisons, add tuned CleanRL/SB3/RL-Zoo runs
or published reference implementations under the same ALE protocol.

## Budget accounting

Do not compare `A-EVIS raw frames` to `PPO agent steps` directly.
With action repeat 4:

```text
raw emulator frames ~= agent decisions * 4
```

A-EVIS and Cold CEM count frames spent in every candidate rollout, including
rejected branches. Whole-program rollouts have variable cost, so the last rollout may
slightly overshoot a requested budget; the implementation counts that entire rollout
rather than silently dropping its cost.

For an approximately 1M raw-frame comparison, the default PPO/DQN reference command
uses 250k agent decisions with action repeat 4.

## Suggested comparison table

For every game and seed, report at minimum:

```text
A-EVIS
Cold CEM (same representation, no library)
PPO
DQN
```

Primary comparison axes:

1. evaluation return vs raw emulator frames;
2. wall-clock time;
3. A-EVIS minus Cold CEM, which isolates amortization;
4. variance across seeds.

## Results policy

`results/atari/README.md` intentionally starts with `PENDING LOCAL VALIDATION`.
When local runs are finished, record:

- exact commit SHA
- package versions
- OS / GPU / CPU
- game, seed, ALE mode/difficulty if changed
- raw frame budget
- wall-clock time
- mean/median episode return and evaluation episodes

Do not replace missing runs with sandbox toy-game results.
