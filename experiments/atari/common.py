"""Shared Atari feature/program utilities for A-EVIS experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtariProgramSpec:
    latent_dim: int = 64
    pooled_size: int = 12
    basis_seed: int = 1729


class AtariLatentProgram:
    """Low-dimensional complete policy hypothesis for pixel Atari.

    The representation is intentionally generic: stacked grayscale frames are
    average-pooled; a fixed random hypernetwork maps a latent vector to a linear
    action controller. A-EVIS searches only the latent vector using complete
    episode outcomes.
    """

    def __init__(self, n_actions: int, frame_stack: int, spec: AtariProgramSpec):
        self.n_actions = int(n_actions)
        self.frame_stack = int(frame_stack)
        self.spec = spec
        self.feature_dim = frame_stack * spec.pooled_size * spec.pooled_size + 1
        rng = np.random.default_rng(spec.basis_seed + 1009 * self.n_actions)
        self.basis = rng.normal(
            0.0,
            1.0 / np.sqrt(spec.latent_dim),
            size=(spec.latent_dim, self.n_actions * self.feature_dim),
        ).astype(np.float32)

    def features(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32) / 255.0
        _, h, w = x.shape
        size = self.spec.pooled_size
        r_edges = np.linspace(0, h, size + 1, dtype=int)
        c_edges = np.linspace(0, w, size + 1, dtype=int)
        pooled = np.empty((x.shape[0], size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                block = x[:, r_edges[i]:r_edges[i + 1], c_edges[j]:c_edges[j + 1]]
                pooled[:, i, j] = block.mean(axis=(1, 2))
        return np.concatenate([pooled.reshape(-1), np.ones(1, np.float32)])

    def weights(self, latent: np.ndarray) -> np.ndarray:
        z = np.asarray(latent, dtype=np.float32).reshape(self.spec.latent_dim)
        flat = z @ self.basis
        return flat.reshape(self.n_actions, self.feature_dim)

    def action(self, latent: np.ndarray, obs: np.ndarray) -> int:
        logits = self.weights(latent) @ self.features(obs)
        return int(np.argmax(logits))

    def fingerprint(self, obs: np.ndarray) -> np.ndarray:
        feats = self.features(obs)[:-1].reshape(
            self.frame_stack,
            self.spec.pooled_size,
            self.spec.pooled_size,
        )
        return feats.mean(axis=0).reshape(-1).astype(np.float32)
