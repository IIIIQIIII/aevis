# Method: A-EVIS

A-EVIS stands for **Amortized Environment-Verified Inductive Search**.

## Operational definition

A-EVIS is defined by four requirements:

1. **The optimization unit is a complete executable program/behavior.**
2. **Candidate quality is measured by executing the complete candidate in an environment.**
3. **No temporal-credit learner is required by the core update.**
4. **Verified successful programs are reused across tasks to amortize future search.**

The prototype intentionally excludes Q/V functions, Bellman/TD updates, policy gradients, advantages, critics, and teacher distillation.

## Current continuous-control instantiation

RobotReach represents a controller as

```text
action_t = tanh(W [observation_t; 1])
```

A program is the complete matrix `W`. Search mutates and evaluates whole matrices. Successful matrices are stored with raw task fingerprints. On later tasks, neighboring verified programs are retrieved and kernel-combined to form a proposal and a local search distribution.

## Why this is not behavior cloning

The library stores no `(state, action)` labels and no teacher trajectories. Reuse happens in **program space**, not by fitting a policy to demonstrated actions.

## Why this is not standard RL

The core update does not construct action values, returns-to-go, advantages, Bellman targets, or score-function gradients. It compares complete executable hypotheses using environment outcomes.

A broad definition of reinforcement learning may still classify any reward-driven adaptation as RL. The project therefore uses the more precise phrase **candidate non-RL learning framework** and states the excluded mechanisms explicitly.

## Open problem: representation

The current program representation is deliberately simple. Generality requires richer representations that can cover discrete games, continuous robotics, and verifier-driven LLM agents without injecting task-specific modules. Program representation is therefore a first-class research problem, not a solved component.
