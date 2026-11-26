"""Scaling plot: parallel envs vs wall-clock time to reach the same reward.

Run:  python mjx/scaling_plot.py logs/rsl_rl/go1_velocity/*_scale_* --target 20
Out:  scaling.png and a printed table (envs, steps/s, minutes to target).
Each run dir must be named ..._scale_<num_envs> (use --agent.run-name scale_<N>).
"""
# [permanent practice] Read TensorBoard event files directly: no dashboard needed for a number.
import argparse
import glob
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    reward = ea.Scalars("Train/mean_reward") if "Train/mean_reward" in tags else []
    fps = [e.value for e in ea.Scalars("Perf/total_fps")] if "Perf/total_fps" in tags else []
    return reward, fps


def minutes_to(reward, target):
    if not reward:
        return None
    t0 = reward[0].wall_time
    for e in reward:
        if e.value >= target:
            return (e.wall_time - t0) / 60
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--target", type=float, required=True, help="Train/mean_reward to reach")
    p.add_argument("--out", default="scaling.png")
    a = p.parse_args()
    rows = []
    for r in sorted(set(sum([glob.glob(x) for x in a.runs], []))):
        m = re.search(r"scale_(\d+)", r)
        if not m:
            continue
        reward, fps = load(r)
        sps = sum(fps) / len(fps) if fps else float("nan")
        rows.append((int(m.group(1)), sps, minutes_to(reward, a.target)))
    rows.sort()
    print(f"{'envs':>6} {'steps/s':>10} {'min to target':>14}")
    for n, sps, mins in rows:
        print(f"{n:>6} {sps:>10.0f} {mins if mins is not None else 'not reached':>14}")
    done = [(n, m) for n, _, m in rows if m is not None]
    if done:
        plt.figure(figsize=(6, 4))
        plt.plot([n for n, _ in done], [m for _, m in done], "o-")
        plt.xscale("log", base=2)
        plt.xlabel("parallel environments")
        plt.ylabel(f"minutes to mean reward {a.target}")
        plt.title("Where more envs stop paying")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(a.out, dpi=150)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
