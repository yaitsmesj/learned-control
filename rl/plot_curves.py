"""Plot evaluation return vs environment steps, mean across seeds with a min-max band.

Run:  python rl/plot_curves.py runs/ppo_CartPole-v1_s* --out ppo_cartpole.png --solved 500
"""
# [permanent practice] Report the deterministic evaluation curve, all seeds, never the best one.
import argparse
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(run_dir):
    with open(f"{run_dir}/progress.csv") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["step"]) for r in rows]), np.array([float(r["eval_return"]) for r in rows])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--solved", type=float, default=None)
    a = p.parse_args()

    curves = [load(r) for r in a.runs]
    n = min(len(s) for s, _ in curves)                  # align on the shortest run
    steps = curves[0][0][:n]
    ys = np.stack([y[:n] for _, y in curves])
    mean, lo, hi = ys.mean(0), ys.min(0), ys.max(0)

    label = re.sub(r"_s\d+$", "", a.runs[0].rstrip("/").split("/")[-1])
    plt.figure(figsize=(7, 4))
    plt.fill_between(steps, lo, hi, alpha=0.25, label=f"min-max over {len(a.runs)} seeds")
    plt.plot(steps, mean, lw=2, label="mean")
    if a.solved is not None:
        plt.axhline(a.solved, ls="--", c="grey", lw=1, label=f"solved ({a.solved:g})")
    plt.xlabel("environment steps"); plt.ylabel("evaluation return (deterministic)")
    plt.title(label); plt.legend(); plt.tight_layout(); plt.savefig(a.out, dpi=150)

    last = ys[:, -1]
    print(f"{label}: final eval return mean {last.mean():.1f}, min {last.min():.1f}, max {last.max():.1f} "
          f"over {len(a.runs)} seeds; per seed {np.round(last, 1).tolist()}")
    if a.solved is not None:
        for r, y in zip(a.runs, ys):
            hit = np.where(y >= a.solved)[0]
            print(f"  {r}: first reached {a.solved:g} at step {int(steps[hit[0]]) if len(hit) else 'never'}")


if __name__ == "__main__":
    main()
