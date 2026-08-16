"""A-EVIS / Cold-EVIS stream experiment on RobotReach.

Run from the repository root after `pip install -e .`:

    python experiments/aevis_robotreach.py --method aevis --seed 0 \
        --budget 60000 --tasks 120 --out /tmp/aevis.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aevis import VerifiedProgramLibrary, cross_entropy_program_search
from benchmarks.robotreach import RobotReachEnv


PROGRAM_SHAPE = (RobotReachEnv.ACTION_DIM, RobotReachEnv.OBS_DIM + 1)


def fingerprint(task_seed: int) -> np.ndarray:
    env = RobotReachEnv(task_seed)
    return np.asarray(env.reset(task_seed), dtype=np.float32)


def controller_action(program: np.ndarray, obs: np.ndarray) -> np.ndarray:
    x = np.concatenate([
        np.asarray(obs, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    ])
    return np.tanh(program @ x).astype(np.float32)


def evaluate_program(task_seed: int, program: np.ndarray) -> tuple[float, bool, int]:
    env = RobotReachEnv(task_seed)
    obs = env.reset(task_seed)
    total_return = 0.0
    interactions = 0

    while True:
        obs, reward, done, info = env.step(controller_action(program, obs))
        total_return += float(reward)
        interactions += 1
        if done:
            return total_return, bool(info["success"]), interactions


def cold_search(
    task_seed: int,
    rng: np.random.Generator,
    interaction_budget: int,
) -> tuple[np.ndarray, float, bool, int]:
    result = cross_entropy_program_search(
        lambda program: evaluate_program(task_seed, program),
        PROGRAM_SHAPE,
        rng,
        interaction_budget,
        population=16,
        elite_count=4,
        generations=3,
    )
    return result.program, result.score, result.verified, result.interactions


def library_search(
    task_seed: int,
    task_fingerprint: np.ndarray,
    library: VerifiedProgramLibrary,
    rng: np.random.Generator,
    interaction_budget: int,
) -> tuple[np.ndarray, float, bool, int]:
    exact = max(1, min(4, interaction_budget // 105 - 1))
    candidates, local_mean, local_std = library.propose(
        task_fingerprint,
        neighbors=9,
        exact_programs=exact,
    )

    used = 0
    best_program = candidates[0]
    best_score = float("-inf")
    best_verified = False

    for candidate in candidates:
        score, verified, cost = evaluate_program(task_seed, candidate)
        if used + cost > interaction_budget and used > 0:
            break
        used += cost
        if score > best_score:
            best_program = candidate.copy()
            best_score = float(score)
            best_verified = bool(verified)
        if verified:
            return best_program, best_score, True, used

    remaining = interaction_budget - used
    if remaining > 180:
        local = cross_entropy_program_search(
            lambda program: evaluate_program(task_seed, program),
            PROGRAM_SHAPE,
            rng,
            remaining,
            init_mean=local_mean,
            init_std=local_std,
            population=max(4, min(10, remaining // 90)),
            elite_count=3,
            generations=1,
        )
        used += local.interactions
        if local.score > best_score:
            best_program = local.program.copy()
            best_score = float(local.score)
            best_verified = bool(local.verified)

    return best_program, best_score, best_verified, used


def run_stream(
    method: str,
    seed: int,
    total_budget: int,
    tasks: int,
    bootstrap_tasks: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    library = VerifiedProgramLibrary(capacity=120)
    total_interactions = 0
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for task_index in range(tasks):
        remaining = total_budget - total_interactions
        tasks_left = tasks - task_index
        task_seed = 90_000_000 + seed * 10_000 + task_index

        if remaining <= 0:
            rows.append({
                "method": method,
                "seed": seed,
                "task_index": task_index,
                "task_seed": task_seed,
                "success": 0,
                "task_interactions": 0,
                "cumulative_interactions": total_interactions,
                "library_size": len(library),
            })
            continue

        task_fingerprint = fingerprint(task_seed)
        use_library = (
            method == "aevis"
            and task_index >= bootstrap_tasks
            and len(library) >= 10
        )

        if use_library:
            task_budget = max(
                100,
                min(remaining, int((remaining / tasks_left) * 1.03)),
            )
            program, score, verified, used = library_search(
                task_seed,
                task_fingerprint,
                library,
                rng,
                task_budget,
            )
        else:
            if method == "aevis":
                task_budget = min(
                    1500,
                    max(200, remaining - max(0, tasks_left - 1) * 260),
                )
            else:
                task_budget = max(
                    100,
                    min(remaining, int(remaining / tasks_left)),
                )
            program, score, verified, used = cold_search(
                task_seed,
                rng,
                task_budget,
            )

        total_interactions += used
        if method == "aevis" and verified:
            library.add(task_fingerprint, program, score)

        rows.append({
            "method": method,
            "seed": seed,
            "task_index": task_index,
            "task_seed": task_seed,
            "success": int(verified),
            "task_interactions": used,
            "cumulative_interactions": total_interactions,
            "library_size": len(library),
        })

    success_rate = float(np.mean([row["success"] for row in rows]))
    summary = {
        "method": method,
        "seed": seed,
        "tasks": tasks,
        "interactions": total_interactions,
        "success_rate": success_rate,
        "successful_tasks": int(sum(row["success"] for row in rows)),
        "mean_interactions_per_task": float(
            np.mean([row["task_interactions"] for row in rows])
        ),
        "final_library_size": len(library),
        "post_bootstrap_success_rate": float(
            np.mean([row["success"] for row in rows[bootstrap_tasks:]])
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["aevis", "cold"], default="aevis")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=60_000)
    parser.add_argument("--tasks", type=int, default=120)
    parser.add_argument("--bootstrap-tasks", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-log")
    args = parser.parse_args()

    summary, rows = run_stream(
        args.method,
        args.seed,
        args.budget,
        args.tasks,
        args.bootstrap_tasks,
    )

    with Path(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    if args.task_log:
        with Path(args.task_log).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(summary)


if __name__ == "__main__":
    main()
