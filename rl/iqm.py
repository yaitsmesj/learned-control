"""Interquartile mean (IQM) of final scores across seeds, with a bootstrap 95% CI.

Run:  python rl/iqm.py runs/ppo_CartPole-v1_s*
Why:  with 3-5 seeds, the mean is dragged by one bad seed and min-max says little.
      IQM drops the top and bottom quarter; the CI says how much to trust it.
      This is what the `rliable` library reports; here it is in 20 lines so you can explain it.
"""
# [permanent practice] Report RL results as IQM + CI over seeds, never one lucky run.
import argparse
import csv
import numpy as np


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return x[lo:hi].mean()


def bootstrap_ci(x, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    stats = [iqm(rng.choice(x, size=len(x), replace=True)) for _ in range(reps)]
    return np.percentile(stats, [2.5, 97.5])


def final_score(run_dir, last_k=5):
    rows = list(csv.DictReader(open(f"{run_dir}/progress.csv")))
    return float(np.mean([float(r["eval_return"]) for r in rows[-last_k:]]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    scores = [final_score(r) for r in a.runs]
    lo, hi = bootstrap_ci(scores)
    print("final scores (mean of last 5 evals):", [round(s, 1) for s in scores])
    print(f"IQM {iqm(scores):.1f}  95% CI [{lo:.1f}, {hi:.1f}]  over {len(scores)} seeds")
