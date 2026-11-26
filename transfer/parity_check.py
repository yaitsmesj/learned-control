"""Parity check: does go1_port.py reproduce what mjlab does, before any transfer number counts?

Checks three things, all against mjlab itself:
  1. config:       timestep, decimation, and the XML PD gains match the ONNX metadata
  2. observations: the 48 numbers the port builds equal the 48 numbers mjlab feeds the policy
  3. one step:     same state + same action -> how far apart are the two simulators after one policy step?
Run in the mjlab env (CPU is fine), from the repo root:
  python transfer/parity_check.py --onnx policy.onnx --scene scene/scene.xml
"""
# [permanent practice] Test the port against the source of truth before trusting any transfer number.
import argparse
import numpy as np
import torch
import mujoco
import mjlab.tasks  # noqa: F401  (registers task IDs)
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from go1_port import Go1Port

p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True)
p.add_argument("--scene", required=True)
p.add_argument("--steps", type=int, default=20)
a = p.parse_args()

cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-Go1", play=True)
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
obs, _ = env.reset()
port = Go1Port(a.scene, a.onnx)
robot = env.scene["robot"]

# 1. config
kp_xml = port.m.actuator_gainprm[port.act_id, 0]
kd_xml = -port.m.actuator_biasprm[port.act_id, 2]
cfg_ok = (abs(port.m.opt.timestep - cfg.sim.mujoco.timestep) < 1e-12
          and port.decimation == cfg.decimation
          and np.allclose(kp_xml, port.meta["kp"], atol=1e-2)
          and np.allclose(kd_xml, port.meta["kd"], atol=1e-2))
print(f"config       timestep {port.m.opt.timestep} vs {cfg.sim.mujoco.timestep}, decimation "
      f"{port.decimation} vs {cfg.decimation}, PD gains match metadata: {cfg_ok}")

# 2. observations (and 3. one step), over random actions
torch.manual_seed(0)
names = ["lin_vel", "ang_vel", "gravity", "joint_pos", "joint_vel", "last_action", "command"]
cuts = np.cumsum([0, 3, 3, 3, 12, 12, 12, 3])
worst_obs = np.zeros(7)
worst_qpos = 0.0


def copy_state():
    port.d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    port.d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(port.m, port.d)


for t in range(a.steps):
    act = 0.3 * (2 * torch.rand(1, 12) - 1)
    copy_state()
    port.apply(act[0].numpy())                      # port: same action through its own mapping
    for _ in range(port.decimation):
        mujoco.mj_step(port.m, port.d)
    obs, *_ = env.step(act)                         # mjlab: same action
    q_mj = env.sim.data.qpos[0].cpu().numpy()
    worst_qpos = max(worst_qpos, np.abs(port.d.qpos[7:] - q_mj[7:]).max())

    copy_state()
    port.last_action = act[0].numpy()
    port.command = env.command_manager.get_command("twist")[0].cpu().numpy()
    mine = port.obs()
    bias = (robot.data.joint_pos_biased - robot.data.joint_pos)[0].cpu().numpy()
    mine[cuts[3]:cuts[4]] += bias  # mjlab adds a per-episode encoder bias; a real robot would too
    ref = obs["actor"][0].cpu().numpy()
    for i in range(7):
        worst_obs[i] = max(worst_obs[i], np.abs(mine[cuts[i]:cuts[i + 1]] - ref[cuts[i]:cuts[i + 1]]).max())

for n, w in zip(names, worst_obs):
    print(f"obs {n:12s} max |port - mjlab| = {w:.2e}")
print(f"one policy step, joint angles: max |port - mjlab| = {worst_qpos:.2e} rad  (MuJoCo vs MuJoCo Warp)")
ok = cfg_ok and worst_obs.max() < 1e-3 and worst_qpos < 1e-2
print("PARITY", "PASS" if ok else "FAIL")
