"""Evaluate the Go1 policy in plain MuJoCo under deliberate shifts.

Run from the repo root (lc env):
  python transfer/shift_eval.py --onnx results/flat_s1.onnx --scene scene/scene.xml --out results/shift_flat_s1.csv
Out: one row per shift: fall rate over episodes and mean velocity-tracking error.
"""
# [permanent robotics] A transfer claim is only as good as the shift table behind it.
import argparse
import csv
from collections import deque
import numpy as np
import mujoco
from go1_port import Go1Port

SHIFTS = {
    "baseline":      {},
    "timestep_2ms":  {"timestep": 0.002, "decimation": 10},   # same 20 ms policy period, finer physics
    "solver_cheap":  {"sim": {"iterations": 3, "ls_iterations": 5}},
    "friction_0.4x": {"friction": 0.4},
    "latency_1step": {"delay": 1},                           # 20 ms
    "latency_2step": {"delay": 2},                           # 40 ms
    "push_1.0mps":   {"push": 1.0},
}
FALL_HEIGHT = 0.15   # m; the trunk stands at ~0.28 m. Tilt alone misses a robot lying flat.
FALL_UP_Z = -0.5     # projected gravity z; -1 when upright, 0 when on its side


def run_episode(scene, onnx_path, shift, seed, seconds=10.0, cmd=(0.5, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    port = Go1Port(scene, onnx_path, timestep=shift.get("timestep", 0.005),
                   decimation=shift.get("decimation", 4), sim=shift.get("sim"))
    if "friction" in shift:
        port.m.geom_friction[:, 0] *= shift["friction"]
    port.reset()
    # Different start per seed, so a "rate" is a rate: perturb joints and heading.
    port.d.qpos[port.qadr] += rng.uniform(-0.05, 0.05, 12)
    yaw = rng.uniform(-np.pi, np.pi)
    port.d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(port.m, port.d)
    port.command = np.array(cmd)
    queue = deque([np.zeros(12)] * shift.get("delay", 0))
    steps = int(round(seconds / (port.m.opt.timestep * port.decimation)))
    errs = []
    for t in range(steps):
        queue.append(port.policy(port.obs()))
        port.apply(queue.popleft())  # the delayed action reaches the motors
        if "push" in shift and t == steps // 2:
            ang = rng.uniform(0, 2 * np.pi)
            port.d.qvel[:2] += shift["push"] * np.array([np.cos(ang), np.sin(ang)])
        for _ in range(port.decimation):
            mujoco.mj_step(port.m, port.d)
        o = port.obs()
        if o[8] > FALL_UP_Z or port.d.qpos[2] < FALL_HEIGHT:
            return True, float(np.mean(errs)) if errs else float("nan")
        errs.append(np.linalg.norm(o[0:2] - port.command[:2]))
    return False, float(np.mean(errs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="shift_results.csv")
    a = p.parse_args()
    rows = []
    for name, shift in SHIFTS.items():
        res = [run_episode(a.scene, a.onnx, shift, seed) for seed in range(a.episodes)]
        fall = float(np.mean([r[0] for r in res]))
        err = float(np.nanmean([r[1] for r in res]))
        rows.append({"shift": name, "fall_rate": fall, "track_err_mps": round(err, 4)})
        print(f"{name:15s} fall {fall:4.0%}  track err {err:.3f} m/s")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
