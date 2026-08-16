"""Whole-program black-box verification search.

This module intentionally exposes a callback-based interface. It does not know
about rewards-to-go, actions, states, Bellman backups, or policy gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class SearchResult:
    program: np.ndarray
    score: float
    verified: bool
    interactions: int


def cross_entropy_program_search(
    evaluate: Callable[[np.ndarray], tuple[float, bool, int]],
    program_shape: tuple[int, ...],
    rng: np.random.Generator,
    interaction_budget: int,
    *,
    init_mean: Optional[np.ndarray] = None,
    init_std: Optional[np.ndarray] = None,
    population: int = 24,
    elite_count: int = 6,
    generations: int = 4,
) -> SearchResult:
    """Search complete programs using only whole-program environment outcomes."""
    mean = (
        np.zeros(program_shape, dtype=np.float32)
        if init_mean is None
        else np.asarray(init_mean, dtype=np.float32).copy()
    )
    std = (
        np.ones(program_shape, dtype=np.float32) * 0.70
        if init_std is None
        else np.clip(np.asarray(init_std, dtype=np.float32), 0.05, 1.20)
    )

    used = 0
    best_program = mean.copy()
    best_score = float("-inf")
    best_verified = False

    for _ in range(generations):
        if used >= interaction_budget:
            break

        half = max(1, population // 2)
        perturbations = rng.normal(size=(half, *program_shape)).astype(np.float32)
        candidates = np.concatenate(
            [mean[None] + std[None] * perturbations, mean[None] - std[None] * perturbations],
            axis=0,
        )[:population]
        candidates[0] = mean

        evaluated: list[tuple[float, bool, np.ndarray]] = []
        for candidate in candidates:
            score, verified, cost = evaluate(candidate)
            if used + cost > interaction_budget and evaluated:
                break
            used += int(cost)
            evaluated.append((float(score), bool(verified), candidate.copy()))

            if score > best_score:
                best_program = candidate.copy()
                best_score = float(score)
                best_verified = bool(verified)

            if verified:
                return SearchResult(candidate.copy(), float(score), True, used)

        if not evaluated:
            break

        evaluated.sort(key=lambda item: item[0], reverse=True)
        elites = np.stack([item[2] for item in evaluated[: min(elite_count, len(evaluated))]])
        new_mean = elites.mean(axis=0)
        new_std = elites.std(axis=0) + 0.05
        mean = 0.35 * mean + 0.65 * new_mean
        std = np.clip(0.35 * std + 0.65 * new_std, 0.05, 1.20)

    return SearchResult(best_program, best_score, best_verified, used)
