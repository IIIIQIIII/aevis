# Prototype results

These results were produced in sandbox experiments and are **not peer reviewed**.

## RobotReach

RobotReach is a self-contained continuous-control benchmark with randomized 2-DoF arm dynamics, target-conditioned control, continuous torques, sensor noise, and cloneable state.

### 120-task stream, ~60k interaction budget, 3 seeds

| Method | Mean interactions | Success rate |
|---|---:|---:|
| A-EVIS | 52,445 | 60.6% ± 4.6% |
| Cold EVIS | 59,302 | 46.7% ± 2.9% |
| TD3 | 60,000 | 48.6% ± 0.5% |
| PPO | 60,000 | 28.1% ± 2.9% |

The strongest causal evidence for amortization is the same-stream ablation:

- seed 0: +15.0 percentage points vs Cold EVIS
- seed 1: +17.5 points
- seed 2: +9.2 points

A-EVIS also used fewer interactions on average.

### Interpretation caveat

A-EVIS and Cold EVIS solve a sequential task stream with per-task verification. TD3/PPO train amortized neural policies and are evaluated under a different deployment pattern. The direct numerical comparison is informative but not a definitive apples-to-apples claim that A-EVIS universally outperforms TD3/PPO.

## What would constitute stronger evidence

Before claiming a new general learning paradigm, the same core algorithm should be tested on:

- a standard game benchmark (e.g. ALE / Atari)
- a standard continuous-control suite (e.g. MuJoCo / Isaac Lab)
- a verifier-driven LLM tool-use/reasoning benchmark
- stronger tuned baselines and more random seeds
