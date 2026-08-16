"""RobotReach: a self-contained Isaac-style continuous-control benchmark."""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Config:
    dt: float = 0.05
    horizon: int = 120
    success_radius: float = 0.10
    hold_steps: int = 3
    sensor_noise: float = 0.0015
    reward_mode: str = "dense"

    def __post_init__(self):
        if self.reward_mode not in ("dense", "sparse"):
            raise ValueError(self.reward_mode)


class RobotReachEnv:
    OBS_DIM = 10
    ACTION_DIM = 2

    def __init__(self, seed=0, config: Optional[Config] = None):
        self.config = config or Config()
        self.rng = random.Random(seed)
        self.reset(seed)

    def clone(self):
        return copy.deepcopy(self)

    def _rand(self, a, b):
        return self.rng.uniform(a, b)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(int(seed))

        self.lengths = np.array([self._rand(.48, .58), self._rand(.38, .48)], np.float64)
        self.inertia = np.array([self._rand(.75, 1.25), self._rand(.45, .85)], np.float64)
        self.damping = np.array([self._rand(.10, .18), self._rand(.07, .14)], np.float64)
        self.gravity = self._rand(.04, .10)
        self.q = np.array([self._rand(-.8, .8), self._rand(-.6, .6)], np.float64)
        self.dq = np.zeros(2, np.float64)

        reach = float(self.lengths.sum())
        for _ in range(100):
            angle = self._rand(-math.pi, math.pi)
            radius = self._rand(.30 * reach, .82 * reach)
            target = np.array([radius * math.cos(angle), radius * math.sin(angle)], np.float64)
            if np.linalg.norm(target - self.end_effector()) > .22:
                self.target = target
                break

        self.steps = 0
        self.hold = 0
        self.done = False
        self.success = False
        self.prev_dist = self.distance()
        return self.obs()

    def joint_positions(self):
        q1, q2 = self.q
        l1, l2 = self.lengths
        p1 = np.array([l1 * math.cos(q1), l1 * math.sin(q1)])
        p2 = p1 + np.array([l2 * math.cos(q1 + q2), l2 * math.sin(q1 + q2)])
        return np.stack([np.zeros(2), p1, p2])

    def end_effector(self):
        return self.joint_positions()[-1].copy()

    def distance(self):
        return float(np.linalg.norm(self.end_effector() - self.target))

    def obs(self):
        qn = self.q + np.array([self.rng.gauss(0, self.config.sensor_noise) for _ in range(2)])
        dqn = self.dq + np.array([self.rng.gauss(0, self.config.sensor_noise) for _ in range(2)])
        ee = self.end_effector()
        out = np.concatenate([np.sin(qn), np.cos(qn), dqn, self.target, ee]).astype(np.float32)
        assert out.shape == (10,)
        return out

    def step(self, action):
        if self.done:
            raise RuntimeError("episode finished")

        u = np.clip(np.asarray(action, np.float64).reshape(2), -1, 1)
        q1, q2 = self.q
        coupling = np.array([
            .05 * math.sin(q2) * self.dq[1],
            .04 * math.sin(q1 - q2) * self.dq[0],
        ])
        gravity = self.gravity * np.array([math.sin(q1), math.sin(q1 + q2)])
        ddq = (u - self.damping * self.dq - gravity - coupling) / self.inertia

        self.dq += self.config.dt * ddq
        self.dq = np.clip(self.dq, -3, 3)
        self.q += self.config.dt * self.dq
        self.q = (self.q + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1

        distance = self.distance()
        self.hold = self.hold + 1 if distance <= self.config.success_radius else 0
        self.success = self.hold >= self.config.hold_steps
        self.done = self.success or self.steps >= self.config.horizon

        if self.config.reward_mode == "sparse":
            reward = 1.0 if self.success else 0.0
        else:
            progress = self.prev_dist - distance
            reward = 5.0 * progress - .002 * float(u @ u) - .0005 * float(self.dq @ self.dq)
            if self.success:
                reward += 1.0

        self.prev_dist = distance
        return self.obs(), float(reward), self.done, {"success": self.success, "distance": distance}


def oracle_action(env: RobotReachEnv):
    """Calibration-only Jacobian controller; not used by A-EVIS."""
    q1, q2 = env.q
    l1, l2 = env.lengths
    theta = np.array([q1, q1 + q2])
    jacobian = np.array([
        [-l1 * math.sin(theta[0]) - l2 * math.sin(theta[1]), -l2 * math.sin(theta[1])],
        [ l1 * math.cos(theta[0]) + l2 * math.cos(theta[1]),  l2 * math.cos(theta[1])],
    ])
    error = env.target - env.end_effector()
    torque = 6.0 * jacobian.T @ error - .8 * env.dq
    return np.clip(torque, -1, 1).astype(np.float32)


def self_test():
    env = RobotReachEnv(123)
    env.reset(123)
    clone = env.clone()
    action = np.array([.2, -.3], np.float32)
    o1, r1, d1, _ = env.step(action)
    o2, r2, d2, _ = clone.step(action)
    assert np.allclose(o1, o2)
    assert abs(r1 - r2) < 1e-10
    assert d1 == d2
    print("RobotReach self-test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(RobotReachEnv(args.seed).obs())


if __name__ == "__main__":
    main()
