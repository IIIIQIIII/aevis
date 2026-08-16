"""Reference PPO/DQN baselines for local standard-Atari validation.

This runner uses Stable-Baselines3 as a convenient reference implementation.
Its purpose is reproducibility, not to claim a state-of-the-art Atari baseline.
Use the same ALE game/config and report both agent steps and estimated raw
emulator frames (agent_steps * action_repeat).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    p.add_argument("--game", default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--agent-steps", type=int, default=250_000)
    p.add_argument("--action-repeat", type=int, default=4)
    p.add_argument("--sticky-actions", type=float, default=0.25)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    try:
        import ale_py
        import gymnasium as gym
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    except Exception as exc:
        raise RuntimeError("Install `pip install -e '.[atari]'` first.") from exc

    gym.register_envs(ale_py)

    def make_one():
        base = gym.make(
            args.game,
            frameskip=1,
            repeat_action_probability=args.sticky_actions,
            full_action_space=False,
        )
        return AtariWrapper(
            base,
            frame_skip=args.action_repeat,
            screen_size=84,
            terminal_on_life_loss=False,
            clip_reward=False,
        )

    env = VecFrameStack(DummyVecEnv([make_one]), n_stack=4)
    cls = PPO if args.algo == "ppo" else DQN
    if args.algo == "ppo":
        model = cls("CnnPolicy", env, seed=args.seed, verbose=1)
    else:
        model = cls(
            "CnnPolicy", env, seed=args.seed, verbose=1,
            learning_starts=20_000, buffer_size=250_000,
        )
    model.learn(total_timesteps=args.agent_steps)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    env.close()

    row = {
        "game": args.game,
        "method": args.algo,
        "seed": args.seed,
        "agent_steps": args.agent_steps,
        "estimated_raw_frames": args.agent_steps * args.action_repeat,
        "eval_episodes": args.eval_episodes,
        "mean_return": float(mean_reward),
        "std_return": float(std_reward),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
