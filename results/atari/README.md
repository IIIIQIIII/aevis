# Atari results — PENDING LOCAL VALIDATION

No standard ALE score is claimed yet.

Use the scripts in `experiments/atari/` and commit locally reproduced CSVs here.
A suggested minimum is 3 seeds for each method/game.

| Game | A-EVIS | Cold CEM | PPO | DQN | Raw-frame budget | Status |
|---|---:|---:|---:|---:|---:|---|
| Pong | — | — | — | — | — | pending |
| Breakout | — | — | — | — | — | pending |
| Seaquest | — | — | — | — | — | pending |
| BeamRider | — | — | — | — | — | pending |

`Cold CEM` uses the same Atari latent program representation and whole-program
search as A-EVIS, but disables the verified-program library. It is the primary
ablation for testing whether amortization helps beyond black-box search alone.
