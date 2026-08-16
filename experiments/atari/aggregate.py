"""Aggregate locally produced Atari CSVs without inventing missing results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="results/atari/summary.csv")
    args = p.parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    data = pd.concat(frames, ignore_index=True, sort=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False)
    print(data.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
