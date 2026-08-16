"""Cold whole-program black-box search baseline for standard Atari.

This uses exactly the same pixel preprocessing, latent program representation,
and CEM whole-program search as A-EVIS, but disables cross-episode verified
program retrieval. It isolates the contribution of A-EVIS amortization from the
contribution of black-box program search itself.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aevis import cross_entropy_program_search
from benchmarks.atari import AtariCloneEnv
from experiments.atari.common import AtariLatentProgram, AtariProgramSpec


def rollout_from_snapshot(env, snapshot, model, latent, max_agent_steps):
    obs = env.restore(snapshot)
    start_frames = env.raw_frames
    score = 0.0
    done = False
    for _ in range(max_agent_steps):
        action = model.action(latent, obs)
        obs, reward, done, _ = env.step(action)
        score += float(reward)
        if done:
            break
    return score, done, env.raw_frames - start_frames


def run(args):
    rng = np.random.default_rng(args.seed)
    env = AtariCloneEnv(
        args.game,
        seed=args.seed,
        action_repeat=args.action_repeat,
        frame_stack=args.frame_stack,
        sticky_actions=args.sticky_actions,
    )
    spec = AtariProgramSpec(
        latent_dim=args.latent_dim,
        pooled_size=args.pooled_size,
        basis_seed=args.basis_seed,
    )
    model = AtariLatentProgram(env.n_actions, args.frame_stack, spec)
    total_frames = 0
    rows = []
    started = time.perf_counter()

    try:
        for episode in range(args.episodes):
            if total_frames >= args.frame_budget:
                break
            obs = env.reset(seed=args.seed + episode)
            root = env.clone()
            remaining = args.frame_budget - total_frames
            local_budget = min(args.frames_per_episode, remaining)

            def evaluate(latent):
                score, done, cost = rollout_from_snapshot(
                    env, root, model, latent, args.max_agent_steps
                )
                return score, False, cost

            result = cross_entropy_program_search(
                evaluate=evaluate,
                program_shape=(args.latent_dim,),
                rng=rng,
                interaction_budget=local_budget,
                population=args.population,
                elite_count=args.elites,
                generations=args.generations,
            )
            total_frames += result.interactions
            rows.append({
                "game": args.game,
                "method": "cold_cem",
                "seed": args.seed,
                "episode": episode,
                "episode_score": result.score,
                "raw_frames_used": result.interactions,
                "cumulative_raw_frames": total_frames,
                "elapsed_s": time.perf_counter() - started,
            })
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out} ({len(rows)} episodes, {total_frames} raw frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--frame-budget", type=int, default=1_000_000)
    p.add_argument("--frames-per-episode", type=int, default=25_000)
    p.add_argument("--max-agent-steps", type=int, default=10_000)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--frame-stack", type=int, default=4)
    p.add_argument("--sticky-actions", type=float, default=0.25)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--pooled-size", type=int, default=12)
    p.add_argument("--basis-seed", type=int, default=1729)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--elites", type=int, default=4)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--out", default="results/atari/cold_pong_seed0.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
