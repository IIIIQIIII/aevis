"""Standard Atari/ALE adapter with exact clone/restore support.

The adapter intentionally keeps Atari-specific work at the environment boundary.
The learning algorithm only sees stacked grayscale observations, a discrete action
space, scalar rewards, terminal flags, and snapshot/restore operations.

Protocol defaults:
- ALE v5 environments
- frameskip=1 inside ALE; action_repeat=4 in this adapter
- repeat_action_probability=0.25 (sticky actions)
- full_action_space=False
- RGB source frames -> max-pool last two -> grayscale -> 84x84 -> stack 4

ROM availability is the user's responsibility; this file never downloads ROMs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class AtariSnapshot:
    ale_state: Any
    frames: tuple[np.ndarray, ...]
    episode_return: float
    agent_steps: int
    raw_frames: int


class AtariCloneEnv:
    def __init__(
        self,
        env_id: str = "ALE/Pong-v5",
        *,
        seed: int = 0,
        action_repeat: int = 4,
        frame_stack: int = 4,
        sticky_actions: float = 0.25,
        screen_size: int = 84,
    ) -> None:
        try:
            import gymnasium as gym
            import ale_py
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Atari support requires `pip install -e '.[atari]'` and locally "
                "available Atari ROMs."
            ) from exc

        gym.register_envs(ale_py)
        self.env_id = env_id
        self.seed = int(seed)
        self.action_repeat = int(action_repeat)
        self.frame_stack = int(frame_stack)
        self.screen_size = int(screen_size)

        # Use the bare Atari environment so clone/restore has no hidden wrapper
        # state. We implement action repeat and frame stacking ourselves.
        wrapped = gym.make(
            env_id,
            obs_type="rgb",
            frameskip=1,
            repeat_action_probability=float(sticky_actions),
            full_action_space=False,
        )
        self._owner = wrapped
        self.env = wrapped.unwrapped
        self.action_space = self.env.action_space
        self.n_actions = int(self.action_space.n)
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.episode_return = 0.0
        self.agent_steps = 0
        self.raw_frames = 0

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32)
        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        return gray.astype(np.uint8)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Atari preprocessing requires opencv-python.") from exc
        return cv2.resize(
            frame,
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)

    def _preprocess(self, previous: np.ndarray, latest: np.ndarray) -> np.ndarray:
        pooled = np.maximum(previous, latest)
        return self._resize(self._gray(pooled))

    def _stacked(self) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("Call reset() before requesting an observation.")
        while len(self._frames) < self.frame_stack:
            self._frames.append(self._frames[-1].copy())
        return np.stack(tuple(self._frames), axis=0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        obs, _ = self.env.reset(seed=self.seed)
        first = self._resize(self._gray(obs))
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(first.copy())
        self.episode_return = 0.0
        self.agent_steps = 0
        self.raw_frames = 0
        return self._stacked()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        reward_sum = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        last_two: list[np.ndarray] = []

        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(int(action))
            reward_sum += float(reward)
            self.raw_frames += 1
            last_two.append(np.asarray(obs))
            if len(last_two) > 2:
                last_two.pop(0)
            if terminated or truncated:
                break

        if len(last_two) == 1:
            last_two.insert(0, last_two[0])
        processed = self._preprocess(last_two[-2], last_two[-1])
        self._frames.append(processed)
        self.agent_steps += 1
        self.episode_return += reward_sum
        done = bool(terminated or truncated)
        return self._stacked(), reward_sum, done, dict(info)

    def clone(self) -> AtariSnapshot:
        ale = self.env.ale
        try:
            state = ale.cloneState(True)
        except TypeError:  # older ALE API
            state = ale.cloneSystemState()
        return AtariSnapshot(
            ale_state=state,
            frames=tuple(frame.copy() for frame in self._frames),
            episode_return=float(self.episode_return),
            agent_steps=int(self.agent_steps),
            raw_frames=int(self.raw_frames),
        )

    def restore(self, snapshot: AtariSnapshot) -> np.ndarray:
        ale = self.env.ale
        if hasattr(ale, "restoreState"):
            ale.restoreState(snapshot.ale_state)
        else:  # pragma: no cover - older ALE API
            ale.restoreSystemState(snapshot.ale_state)
        self._frames = deque(
            (frame.copy() for frame in snapshot.frames),
            maxlen=self.frame_stack,
        )
        self.episode_return = float(snapshot.episode_return)
        self.agent_steps = int(snapshot.agent_steps)
        self.raw_frames = int(snapshot.raw_frames)
        return self._stacked()

    def close(self) -> None:
        self._owner.close()
