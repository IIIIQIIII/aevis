"""Task-agnostic verified-program retrieval and proposal utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProgramEntry:
    """A complete program that has been verified by an environment."""

    fingerprint: np.ndarray
    program: np.ndarray
    score: float


class VerifiedProgramLibrary:
    """Nonparametric archive of environment-verified complete programs.

    The library stores no state/action value targets. Retrieval operates on task
    fingerprints, and proposals combine complete programs in program space.
    """

    def __init__(self, capacity: int = 256):
        self.capacity = int(capacity)
        self.entries: list[ProgramEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, fingerprint: np.ndarray, program: np.ndarray, score: float) -> None:
        self.entries.append(
            ProgramEntry(
                fingerprint=np.asarray(fingerprint, dtype=np.float32).copy(),
                program=np.asarray(program, dtype=np.float32).copy(),
                score=float(score),
            )
        )
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity :]

    def propose(
        self,
        fingerprint: np.ndarray,
        *,
        neighbors: int = 9,
        exact_programs: int = 4,
        scale_floor: float = 0.05,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """Retrieve programs and return candidates plus a local search prior."""
        if not self.entries:
            raise RuntimeError("Cannot propose from an empty verified library.")

        fingerprints = np.stack([entry.fingerprint for entry in self.entries])
        programs = np.stack([entry.program for entry in self.entries])
        query = np.asarray(fingerprint, dtype=np.float32)

        scale = fingerprints.std(axis=0) + float(scale_floor)
        distances = np.mean(((fingerprints - query) / scale) ** 2, axis=1)
        indices = np.argsort(distances)[: min(neighbors, len(self.entries))]
        local_distances = distances[indices]

        bandwidth = float(np.median(local_distances)) + 1e-6
        weights = np.exp(-local_distances / bandwidth)
        weights /= weights.sum()

        local_programs = programs[indices]
        mean_program = np.tensordot(weights, local_programs, axes=(0, 0)).astype(np.float32)
        local_variance = np.tensordot(
            weights,
            (local_programs - mean_program) ** 2,
            axes=(0, 0),
        )
        std_program = np.clip(np.sqrt(local_variance) + 0.05, 0.05, 1.0).astype(np.float32)

        candidates: list[np.ndarray] = [mean_program]
        candidates.extend(
            self.entries[int(index)].program
            for index in indices[: min(exact_programs, len(indices))]
        )

        unique: list[np.ndarray] = []
        for candidate in candidates:
            if not any(np.allclose(candidate, previous, atol=1e-7) for previous in unique):
                unique.append(np.asarray(candidate, dtype=np.float32).copy())

        return unique, mean_program, std_program
