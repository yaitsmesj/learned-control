# Learned Control

One Go1 walking policy, three places: trained with PPO on a free GPU in mjlab, checked in plain MuJoCo under deliberate physics shifts, and driven over ROS 2 from `/cmd_vel`. Before that, PPO and SAC written from scratch and checked against a reference implementation.

Sim-to-sim only. No hardware, no sim2real claim.

## Results

| Study | Number | Picture |
|---|---|---|
| PPO from scratch, CartPole, 5 seeds | (IQM + CI) | `runs/ppo_cartpole.png` |
| SAC from scratch, Pendulum, 3 seeds | (IQM + CI) | `runs/sac_pendulum.png` |
| Scaling: envs vs time-to-reward | (knee at N envs) | `scaling.png` |
| Reward cheating log | (versions) | `docs/reward_log.md` |
| DR ablation | (table) | `docs/dr_ablation.md` |
| Transfer shift table | (fall rate per shift) | `transfer/shift_results.csv` |

## Run it

```bash
uv sync --group dev
uv run pytest -q
uv run python rl/ppo.py --seed 1
```

## What is tested

See the Executed vs Desk-checked ledger in `docs/ledger.md`.
