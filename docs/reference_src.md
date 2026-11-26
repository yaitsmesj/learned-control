# The Learned Control Reference

**Code, commands, configs and fixes for the Learned Control build.**
Companion to *The Learned Control Build Manual*. The Manual says what to do and why; this file holds everything you copy, run or look up.
Every listing below is the exact file from the tested repo, pasted in by a script, so the code here and the code that ran cannot drift apart.
Edition 1 · 22 November 2026

---

*Companion document: **The Learned Control Build Manual**.*

## Contents

- [R0: Repo layout and the four environments](#r0-repo-layout-and-the-four-environments)
- [R1: PPO from scratch](#r1-ppo-from-scratch)
- [R2: SAC from scratch](#r2-sac-from-scratch)
- [R3: Tests](#r3-tests)
- [R4: Plots and statistics](#r4-plots-and-statistics)
- [R5: The CleanRL reference run](#r5-the-cleanrl-reference-run)
- [R6: mjlab: gate, train, resume, read](#r6-mjlab-gate-train-resume-read)
- [R7: The Go1 task, as configured](#r7-the-go1-task-as-configured)
- [R8: Ablation variants and the scaling plot](#r8-ablation-variants-and-the-scaling-plot)
- [R9: Transfer: the plain-MuJoCo port](#r9-transfer-the-plain-mujoco-port)
- [R10: The ROS 2 node](#r10-the-ros-2-node)
- [R11: LeRobot stretch: Diffusion Policy on PushT](#r11-lerobot-stretch-diffusion-policy-on-pusht)
- [R12: Repo hygiene: README, CI, NOW, LOGBOOK](#r12-repo-hygiene-readme-ci-now-logbook)
- [R13: Compute budget](#r13-compute-budget)
- [R14: Troubleshooting](#r14-troubleshooting)
- [R15: Versions](#r15-versions)
- [R16: Test ledger: Executed vs Desk-checked](#r16-test-ledger-executed-vs-desk-checked)

---

## R0: Repo layout and the four environments

```text
learned-control/
├── pyproject.toml          # Phase A + transfer env (Python 3.12, uv)
├── rl/                     # ppo.py, sac.py, plot_curves.py, iqm.py
├── tests/                  # test_math.py: the formulas, pinned
├── mjx/                    # gate_t4.sh, variants.py, scaling_plot.py   (mjlab env)
├── transfer/               # go1_port.py, parity_check.py, shift_eval.py, go1_ros_node.py
├── scene/                  # exported Go1 scene.xml + assets/ meshes
├── runs/                   # Phase A: progress.csv + plots per run (model.pt ignored)
├── results/                # Phase B–C evidence you commit: params/, tfevents, .onnx, shift CSVs
├── docs/                   # reward_log.md, dr_ablation.md, ledger.md (yours) · reference_src.md + build_reference.py
├── NOW.md · LOGBOOK.md · README.md
└── .github/workflows/ci.yml
```

Four environments, kept apart on purpose. Mixing them is the most common way to lose a day.

| Env | Where | Python | What runs in it | Make it |
|---|---|---|---|---|
| `lc` | Mac + here | 3.12 | `rl/`, `tests/`, `transfer/` (not the ROS node); MuJoCo pinned to 3.11.0, mjlab's version | `uv sync --group dev` |
| `mjlab` | Kaggle T4 (train) · Mac (play, parity, scene export) | 3.12 | `train`, `play`, `export-scene`, `mjx/`, `transfer/parity_check.py` | `uv venv -p 3.12 .venv-mjlab && uv pip install -p .venv-mjlab mjlab==1.6.1 onnxruntime tensorboard` |
| `cleanrl` | Mac + here | 3.10 | the reference PPO only | see R5 |
| `ros` | Mac | 3.12 | `transfer/go1_ros_node.py` | RoboStack pixi, see R10 |

<<<FILE pyproject.toml toml>>>

**Why `pythonpath = ["."]`:** tests import `rl.ppo` as a package from the repo root. That is the standard pytest setting, and it is why there is no `sys.path` hack anywhere.

---

## R1: PPO from scratch

Discrete actions, CartPole-v1, one file. Read it top to bottom once; the four numbered comments in `main()` are the whole algorithm.

<<<FILE rl/ppo.py python>>>

**Run (5 seeds, two at a time, one thread each):**

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in 1 3 5; do uv run python rl/ppo.py --seed $s; done &
for s in 2 4;   do uv run python rl/ppo.py --seed $s; done &
wait
```

About 2 minutes per seed on a 2-core machine with one thread each. Five processes with default threading took over 3 minutes for the first 10k steps — see R14.

**Lines worth knowing by heart** (these are the OWN items):

| Line | What it does | What breaks without it |
|---|---|---|
| `autoreset_mode=SAME_STEP` | the obs after a done is already the next episode's; the real last obs is in `info["final_obs"]` | in the 1.x default (`NEXT_STEP`) the step after a done is a dummy transition you must mask |
| `r[i] += gamma * V(final_obs)` on `trunc & ~term` | a time-out keeps its future value | the critic learns "step 500 is a cliff" |
| `done_buf = term \| trunc` | GAE never reaches across an episode boundary | the double-bootstrap bug in M1 |
| per-minibatch advantage normalisation | keeps the policy step size stable across reward scales | updates swing with reward magnitude |
| `ratio.clamp(1-clip, 1+clip)` inside `min` | the clipped surrogate | one big batch can wreck the policy |
| `clip_grad_norm_(..., 0.5)` | caps any single update | rare exploding steps |

---

## R2: SAC from scratch

Continuous actions, Pendulum-v1, same layout as PPO so the two read side by side.

<<<FILE rl/sac.py python>>>

**Run:**

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in 1 2 3; do uv run python rl/sac.py --seed $s & done; wait
```

About 15 minutes for three seeds in parallel on 2 cores (20k steps each, one gradient step per env step).

| Line | What it does | What breaks without it |
|---|---|---|
| `buf.add(..., float(term))` | store **terminated only** | time-outs teach "the future is zero here" |
| `min(q1_t, q2_t)` in the target | twin critics, pessimistic target | Q over-estimates, actor chases phantom value |
| `logp -= log(scale*(1-y²)+1e-6)` | tanh change-of-variables | entropy term is wrong, α tunes to nonsense |
| `target_entropy = -act_dim` | auto-α target | exploration never settles |
| `soft_update(τ=0.005)` | slow-moving target nets | the target moves as fast as the critic and diverges |
| `learning_starts = 5000` | random actions fill the buffer first | early updates overfit a few hundred samples |

---

## R3: Tests

<<<FILE tests/test_math.py python>>>

```bash
uv run pytest -q        # 5 passed
```

**Rule:** every bug you fix gets a test that would have caught it. `test_timeout_does_not_double_bootstrap` is the first one.

---

## R4: Plots and statistics

<<<FILE rl/plot_curves.py python>>>

<<<FILE rl/iqm.py python>>>

```bash
uv run python rl/plot_curves.py runs/ppo_CartPole-v1_s{1,2,3,4,5} --out runs/ppo_cartpole.png --solved 475
uv run python rl/iqm.py runs/ppo_CartPole-v1_s{1,2,3,4,5}
uv run python rl/plot_curves.py runs/sac_Pendulum-v1_s* --out runs/sac_pendulum.png --solved -200
uv run python rl/iqm.py runs/sac_Pendulum-v1_s*
```

**Evaluation protocol (use it everywhere, including Phase C):** separate eval env · fixed eval seeds (10 000+) · deterministic actions (argmax / tanh(mean)) · report IQM and 95% bootstrap CI over seeds · plot mean with a min–max band · never quote the best seed alone.

---

## R5: The CleanRL reference run

CleanRL's `ppo.py` (commit `fe8d8a0`, April 2026) still uses the pre-1.x Gymnasium API, so it needs its own environment. It is the yardstick for your PPO, not a dependency.

```bash
git clone https://github.com/vwxyzjn/cleanrl ~/cleanrl && cd ~/cleanrl
uv venv -p 3.10 && source .venv/bin/activate
uv pip install "gymnasium==0.29.1" "torch==2.4.1" tensorboard tyro "numpy<2"
for s in 1 2 3; do python cleanrl/ppo.py --seed $s --total-timesteps 500000 & done; wait
tensorboard --logdir runs
```

Tags to compare with your CSV: `charts/episodic_return` ↔ `train_return`, `losses/explained_variance`, `losses/approx_kl`, `losses/clipfrac`, `losses/entropy`, `losses/value_loss`.

---

## R6: mjlab: gate, train, resume, read

mjlab trains only on an NVIDIA GPU. A Mac can `play` and run the parity check on CPU (Warp's CPU backend). The CPU smoke run here managed **18 steps/s with 4 envs** — fine for checking plumbing, useless for training.

### The Day-1 gate

<<<FILE mjx/gate_t4.sh bash>>>

Fill this in from the gate logs; every budget number later depends on it.

| | 1024 envs | 4096 envs |
|---|---|---|
| Steps per second | | |
| Seconds per iteration | | |
| Peak GPU memory (MB) | | |
| Fits a 16 GB T4? | | |

### The Kaggle loop

Every GPU run is a **committed notebook** ("Save & Run All"), not an interactive session: it runs headless until it finishes or hits the session limit, and its `/kaggle/working` is kept as the version's output.

```text
# notebook cells
!git clone https://github.com/<you>/learned-control
%cd learned-control
!pip install -q "mjlab==1.6.1" onnxruntime tensorboard
!train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs 4096 --agent.max-iterations 1500 \
    --agent.logger tensorboard --agent.seed 1 --agent.run-name flat_s1
!cp -r logs /kaggle/working/
```

```bash
# on the Mac: pull the output of the latest version
kaggle kernels output <you>/<notebook-slug> -p kaggle_out/
# keep what the write-up needs
mkdir -p results/flat_s1 && cp -r kaggle_out/logs/rsl_rl/go1_velocity/*_flat_s1/{params,*.onnx,events.out.tfevents.*} results/flat_s1/
```

**Resume after a session died:** upload the run dir as a Kaggle Dataset (`kaggle datasets create -p <dir>` the first time, `kaggle datasets version -p <dir> -m "..."` after), attach it to the notebook, then copy it into place before training:

```text
!mkdir -p logs/rsl_rl/go1_velocity && cp -r /kaggle/input/<dataset>/<run-dir> logs/rsl_rl/go1_velocity/
!train ... --agent.resume True --agent.load-run <run-dir> --agent.load-checkpoint model_1450.pt
```

### Train, play, resume

```bash
# train (Kaggle; logs land in ./logs/rsl_rl/go1_velocity/<timestamp>_<run-name>/)
train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs 4096 \
  --agent.max-iterations 1500 --agent.logger tensorboard \
  --agent.seed 1 --agent.run-name flat_s1

# both T4s on Kaggle's "GPU T4 x2"
train Mjlab-Velocity-Flat-Unitree-Go1 --gpu-ids "[0, 1]" --env.scene.num-envs 4096 ...

# resume after a Kaggle session dies: copy the old run dir back under ./logs/rsl_rl/go1_velocity/ first
train Mjlab-Velocity-Flat-Unitree-Go1 --agent.resume True \
  --agent.load-run <timestamp>_flat_s1 --agent.load-checkpoint model_1450.pt ...

# watch a checkpoint (Mac, CPU). viser opens in the browser; the native viewer needs mjpython on macOS
play Mjlab-Velocity-Flat-Unitree-Go1 --checkpoint-file results/flat_s1/model_1500.pt --viewer viser

# sanity-check the task with no learning
play Mjlab-Velocity-Flat-Unitree-Go1 --agent zero
play Mjlab-Velocity-Flat-Unitree-Go1 --agent random

# record a video on a headless box (Kaggle): offscreen rendering needs EGL
MUJOCO_GL=egl play Mjlab-Velocity-Flat-Unitree-Go1 --checkpoint-file <...>.pt --video True --video-length 400
```

### What a run directory holds

| File | What it is |
|---|---|
| `model_<iter>.pt` | checkpoint every `save_interval` (default 50) iterations |
| `<timestamp>_<run-name>.onnx` | the actor, exported automatically, **with the observation normaliser baked in** (first ops are `Sub`, `Div`) and metadata: joint names, default pose, action scale, PD gains, observation names |
| `params/env.yaml` · `params/agent.yaml` | the exact config used — this is what you diff between ablations |
| `events.out.tfevents.*` | TensorBoard scalars |

### TensorBoard tags worth watching

| Tag | Healthy direction |
|---|---|
| `Train/mean_reward` | up, then flat |
| `Train/mean_episode_length` | up toward 1000 (20 s ÷ 0.02 s) as falls stop |
| `Episode_Reward/<term>` (prefix may include `Episode/`) | which reward term is doing the work |
| `Metrics/twist/error_vel_xy` | down: velocity tracking error |
| `Episode_Termination/fell_over` | down |
| `Policy/mean_std` | slowly down; flat at 1.0 means nothing is learning |
| `Perf/total_fps` | throughput; reported alongside the scaling plot, not instead of it |

The `Train/*` and `Episode_*` tags appear only once episodes finish, so a 2-iteration smoke run shows only `Loss/*`, `Perf/*`, `Policy/*`, `Metrics/*`.

**Kaggle notes:** turn Internet on (needs a phone-verified account). Only `/kaggle/working` of a committed version survives. Assume any run can die: checkpoints every 50 iterations, and the resume path above. Check the current weekly GPU quota and session limit on Kaggle's settings page; they change.

---

## R7: The Go1 task, as configured

Read from mjlab 1.6.1's effective config, not from memory.

| Item | `Mjlab-Velocity-Flat-Unitree-Go1` |
|---|---|
| Physics step · decimation · policy rate | 0.005 s · 4 · 50 Hz (0.02 s) |
| Solver (set at runtime, **not** in the exported XML) | implicit-fast integrator · elliptic cone · impratio 10 · Newton, 10 iterations, 20 line-search iterations, ls tolerance 0.01 · ccd 50 |
| Episode | 20 s → 1000 policy steps |
| Actor obs (48) | `base_lin_vel` 3 (velocimeter site: simulator truth; a real robot must estimate it) · `base_ang_vel` 3 (IMU gyro) · `projected_gravity` 3 · `joint_pos` 12 (relative to default, plus encoder bias) · `joint_vel` 12 · `actions` 12 (last action) · `command` 3 (vx, vy, wz) |
| Obs noise in training | lin vel ±0.5, ang vel ±0.2, gravity ±0.05, joint pos ±0.01, joint vel ±1.5 (uniform); off in play mode |
| Critic extras | true joint pos, foot height, air time, contact, contact forces |
| Action (12) | joint position target = default pose + scale × action; scale 0.373 hip/thigh, 0.249 calf |
| Actuators | PD position: kp 15.9 / kd 1.01 hip+thigh, kp 35.8 / kd 2.28 calf; effort 23.7 / 35.55 N·m |
| Default pose | hip ±0.1, thigh 0.9, calf −1.8 rad |
| Rewards (weight) | track_linear_velocity 2.0 · track_angular_velocity 2.0 · upright 1.0 · pose 1.0 · dof_pos_limits −1.0 · action_rate_l2 −0.1 · foot_clearance −2.0 · foot_swing_height −0.25 · foot_slip −0.1 · soft_landing −1e-5 · (body_ang_vel, angular_momentum, air_time at 0) |
| Events (DR) | push_robot every 1–3 s (±0.5 m/s x,y) · encoder_bias · base_com · foot_friction_slide (0.3–1.5) / spin / roll at startup |
| Terminations | time_out · fell_over |
| Commands | vx, vy ∈ ±1.0 m/s, wz ∈ ±0.5 rad/s. `command_vel` curriculum widens vx to (−1.5, 2.0), wz to ±0.7 at 120 000 policy steps (iteration 5000), and vx to (−2.0, 3.0) at 240 000 (iteration 10 000) |
| Rough adds | terrain, `terrain_levels` curriculum, height scan in obs, collision penalties (self, shank, trunk/head −0.1) |
| PPO (rsl_rl 5.4.2) | 24 steps/env · 5 epochs · 4 minibatches · lr 1e-3 adaptive (desired KL 0.01) · entropy 0.01 · clip 0.2 · obs normalisation on · MLP 512-256-128 ELU · max 10 000 iterations · save every 50 |

Reference curves to compare against: mjlab's nightly benchmarks at `https://mujocolab.github.io/mjlab/nightly/`.

---

## R8: Ablation variants and the scaling plot

**Simple overrides work from the CLI** (verified):

```bash
train Mjlab-Velocity-Flat-Unitree-Go1 --env.rewards.foot-slip.weight 0.0 ...
train Mjlab-Velocity-Flat-Unitree-Go1 --env.rewards.action-rate-l2.weight 0.0 ...
```

**Tuple-valued event params do not** (`--env.events.push-robot.params.velocity-range.x 0.0 0.0` fails to parse in mjlab 1.6.1 — see R14). Structural changes go through named variants instead:

<<<FILE mjx/variants.py python>>>

```bash
python mjx/variants.py Lc-Go1-Flat-NoDR --env.scene.num-envs 4096 --agent.run-name nodr_s1 ...
```

Registered and checked (each variant's events listed from its effective config): `Lc-Go1-Flat-NoDR`, `Lc-Go1-Flat-NoPush`, `Lc-Go1-Flat-NoEncoderBias`, `Lc-Go1-Flat-NoBaseCom`, `Lc-Go1-Flat-NoFootFriction`, `Lc-Go1-Rough-NoCurriculum`. A new axis is one `VARIANTS` line.

<<<FILE mjx/scaling_plot.py python>>>

```bash
# Same env-step budget B for every N, so the comparison is fair: iterations = B / (24 * N).
# Example B = 50M steps: 4096 → 509 iterations, 1024 → 2035, 256 → 8138, 64 → 32552 (likely not reached in time: record it).
for N in 64 256 1024 4096; do
  train Mjlab-Velocity-Flat-Unitree-Go1 --env.scene.num-envs $N \
    --agent.max-iterations $(( 50000000 / (24 * N) )) --agent.logger tensorboard --agent.run-name scale_$N
done
python mjx/scaling_plot.py logs/rsl_rl/go1_velocity/*_scale_* --target <reward all runs can reach>
```

---

## R9: Transfer: the plain-MuJoCo port

**Export the scene once (mjlab env), then copy the meshes** — `export-scene` writes the XML but not the STL files:

```bash
export-scene Mjlab-Velocity-Flat-Unitree-Go1 --output-dir scene
cp -r .venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_go1/xmls/assets scene/
```

Three traps the port must handle (all found here, all in the code):

| Trap | Exported XML | What mjlab actually runs |
|---|---|---|
| Timestep | `0.002` | `0.005` (set at runtime) |
| Solver settings | Euler · pyramidal cone · impratio 1 · 100 iterations | implicit-fast · elliptic · impratio 10 · 10 Newton iterations (R7) |
| Actuator order | FR_hip, FR_thigh, FL_hip, FL_thigh, …, then the four calves | policy order is FR_hip, FR_thigh, FR_calf, FL_hip, … (from ONNX metadata) |

<<<FILE transfer/go1_port.py python>>>

<<<FILE transfer/parity_check.py python>>>

```bash
# mjlab env, CPU is fine, from the repo root
python transfer/parity_check.py --onnx results/flat_s1/<stamp>_flat_s1.onnx --scene scene/scene.xml
```

Executed output (untrained smoke-run policy, 20 random-action steps):

```text
config       timestep 0.005 vs 0.005, decimation 4 vs 4, PD gains match metadata: True
obs lin_vel      max |port - mjlab| = 5.96e-08
obs ang_vel      max |port - mjlab| = 1.79e-07
obs gravity      max |port - mjlab| = 2.38e-07
obs joint_pos    max |port - mjlab| = 4.84e-08
obs joint_vel    max |port - mjlab| = 0.00e+00
obs last_action  max |port - mjlab| = 0.00e+00
obs command      max |port - mjlab| = 0.00e+00
one policy step, joint angles: max |port - mjlab| = 4.73e-03 rad  (MuJoCo vs MuJoCo Warp)
PARITY PASS
```

**What parity does not prove:** that the two simulators stay close over many steps, or that contacts behave the same. That is what the shift table's baseline row measures.

<<<FILE transfer/shift_eval.py python>>>

```bash
# lc env (plain MuJoCo 3.11.0), on the Mac, from the repo root; one CSV per policy
uv run python transfer/shift_eval.py --onnx results/flat_s1/<stamp>_flat_s1.onnx \
  --scene scene/scene.xml --episodes 20 --out results/shift_flat_s1.csv
```

---

## R10: The ROS 2 node

<<<FILE transfer/go1_ros_node.py python>>>

**Env (RoboStack, same pins as Project 1):**

```toml
# pixi.toml additions
[dependencies]
python = "3.12.*"
ros-humble-desktop = "*"
ros-humble-teleop-twist-keyboard = "*"

[pypi-dependencies]
mujoco = "==3.11.0"
onnx = "*"
onnxruntime = "*"
```

```bash
pixi shell
python transfer/go1_ros_node.py --ros-args -p onnx:=results/flat_s1/<stamp>_flat_s1.onnx -p scene:=scene/scene.xml
# second terminal
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}}"
ros2 topic hz /odom                       # expect ~50 Hz
ros2 topic echo /odom --field twist.twist.linear
ros2 run teleop_twist_keyboard teleop_twist_keyboard
rviz2    # Fixed Frame: odom · add TF and Odometry (/odom) displays
```

---

## R11: LeRobot stretch: Diffusion Policy on PushT

LeRobot 0.7.0 (PyPI, 5 Nov 2026), Python ≥ 3.12. Run on Kaggle (GPU) or Colab.

```bash
pip install "lerobot[training,pusht,diffusion]==0.7.0"

lerobot-train \
  --policy.type=diffusion --env.type=pusht --dataset.repo_id=lerobot/pusht \
  --output_dir=outputs/train/dp_pusht --job_name=dp_pusht \
  --steps=50000 --batch_size=64 --save_freq=10000 --env_eval_freq=10000 \
  --policy.device=cuda --policy.push_to_hub=false

lerobot-eval \
  --policy.path=outputs/train/dp_pusht/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
```

Two flags that bite in 0.7.0: `--policy.push_to_hub` defaults to **true** (training refuses to start without a `repo_id` unless you set it false), and the eval interval is `--env_eval_freq` (not `eval_freq`). Alternative task: `--policy.type=act --env.type=aloha` with `lerobot[aloha]` and an ALOHA insertion dataset.

---

## R12: Repo hygiene: README, CI, NOW, LOGBOOK

<<<FILE README.md markdown>>>

<<<FILE .github/workflows/ci.yml yaml>>>

<<<FILE NOW.md markdown>>>

<<<FILE LOGBOOK.md markdown>>>

<<<FILE .gitignore text>>>

---

## R13: Compute budget

**The formula:** GPU-hours for a run = iterations × 24 × envs ÷ (steps per second from the gate) ÷ 3600.
Example: 1500 iterations at 4096 envs = 147M steps. At 20 000 steps/s that is 2.0 h; at 40 000, 1.0 h. The gate tells you which.

Planning assumption until Day 2: **2 GPU-h per full run** (1500 iterations at 4096 envs), 0.5 h per short run.

| Study | Runs | GPU-h each | GPU-h |
|---|---|---|---|
| Gate | 2 short | 0.25 | 0.5 |
| Flat baseline | 3 seeds (2 if tight) | 2 | 6 |
| Scaling plot (64 / 256 / 1024 / 4096), 50M steps each | 4 | ≤ 1 | 4 |
| Reward-cheating log (v0–v5, 500 iterations) | 6 short | 0.7 | 4 |
| Curriculum: Rough with vs without | 2 | 2.5 | 5 |
| DR ablation: 5 Flat variants, 1 seed each | 5 | 2 | 10 |
| Re-runs and mistakes (+25%) | | | 7 |
| **Total, weeks 3–5** | | | **~37** |
| Diffusion Policy on PushT (stretch, week 7) | 1 | 4 | 4 |

That is ~12 GPU-h a week for three weeks. Kaggle's free tier has historically been ~30 GPU-h a week; check the live number. Everything else — Phase A, parity, shift evaluation, ROS — is CPU and free.

**Cut order if the gate says it doesn't fit:** flat baseline to 2 seeds · drop the 64-env scaling point · DR variants at 2048 envs · curriculum pair at 1000 iterations · Diffusion Policy to Colab.

---

## R14: Troubleshooting

| Symptom | Cause | Check / fix |
|---|---|---|
| `train_return` is `nan` in your PPO | Gymnasium 1.x SAME_STEP puts episode stats in `info["final_info"]["episode"]` with an `_episode` mask | read from `final_info`, as `ppo.py` does |
| CleanRL `ppo.py` crashes on `final_info` | it uses the pre-1.x API | its own env, R5 |
| Five training runs crawl (10k steps in 3+ min) | each PyTorch process grabs every core | `OMP_NUM_THREADS=1`, run ≤ cores processes at a time |
| PPO hits 500 then sags to 200–450 late | **open incident** — see M1 in the Manual | compare with CleanRL one change at a time |
| Explained variance ≈ 0 while return is 500 | returns are nearly constant (every episode 500), so their variance is tiny | normal late in CartPole; judge EV early, and on harder tasks |
| SAC eval stuck near −1200 after 5k steps | learning hasn't started or α collapsed | `learning_starts`, watch `alpha` column (should fall from 0.2 to ~0.02) |
| `mjlab` train on Mac fails | no NVIDIA GPU | train on Kaggle; Mac is `play` and parity only |
| `Unrecognized options: 0.0, 0.0` / `does not match expected type Tuple[float, ...]` | tuple-valued event params don't parse from the CLI in mjlab 1.6.1 | use `mjx/variants.py`; good upstream issue/PR candidate |
| `Error opening file 'assets/trunk.stl'` | `export-scene` doesn't copy meshes | copy `asset_zoo/robots/unitree_go1/xmls/assets` next to `scene.xml` |
| Port walks badly though training looked fine | exported XML's timestep, solver settings or actuator order | `parity_check.py` must PASS before any shift number counts |
| `shift_eval` says 0% falls for a policy that obviously collapses | tilt-only fall check (old version) | current version also checks trunk height < 0.15 m |
| Rough policy won't load in the port | Rough adds a height scan: input is no longer 48 | score Flat policies only |
| `play --viewer native` fails on macOS | native viewer needs `mjpython` | `--viewer viser` |
| `--video` fails on Kaggle | no display | `MUJOCO_GL=egl` |
| `torch.cuda.is_available()` False on Kaggle after installing mjlab | pip replaced the CUDA build of torch | read `gate/torch.txt`; reinstall torch for Kaggle's CUDA |
| Kaggle run output missing | ran interactively | "Save & Run All (commit)"; `kaggle kernels output` |
| Resume starts from scratch | run dir not under `./logs/rsl_rl/go1_velocity/` in this session | copy it back, then `--agent.resume True --agent.load-run <dir>` |
| No `Train/mean_reward` tag | no episode has finished yet | wait ~1000 policy steps per env |
| `'repo_id' argument missing` from `lerobot-train` | `push_to_hub` defaults to true | `--policy.push_to_hub=false` |
| huggingface.co blocked | network policy | run LeRobot on Kaggle/Colab |

---

## R15: Versions

Checked 22 November 2026.

| Package | Version | Where |
|---|---|---|
| Python | 3.12 | all but CleanRL (3.10) |
| gymnasium | 1.3.0 | lc |
| torch | 2.15.0 (lc, mjlab) · 2.4.1 (CleanRL) | |
| mujoco (standalone) | 3.11.0, pinned to match mjlab (latest on PyPI: 4.0.1) | lc |
| mjlab | 1.6.1 (PyPI) | mjlab |
| mujoco / mujoco-warp (inside mjlab) | 3.11.0 / 3.11.0 | mjlab |
| warp-lang | 1.19.0 | mjlab |
| rsl-rl-lib | 5.4.2 | mjlab |
| tyro | 1.0.17 | mjlab |
| onnx / onnxruntime | latest | lc, mjlab |
| CleanRL | commit `fe8d8a0` | cleanrl |
| LeRobot | 0.7.0 | Kaggle/Colab |
| ROS 2 | Humble via RoboStack (pixi) | ros |

---

## R16: Test ledger: Executed vs Desk-checked

**Executed** = run here and the output shown is real. **Desk-checked** = written against the source and docs, not run, because this machine can't (no GPU, Hugging Face and RoboStack blocked).

| Item | Status | Evidence |
|---|---|---|
| `tests/test_math.py` (5 tests) | Executed | 5 passed |
| CI steps (`uv sync`, pytest, 2k-step PPO) in a clean dir | Executed | passed |
| PPO, 5 seeds × 500k | Executed | all reach 475 by 40–90k steps; final IQM 407 [267, 500] |
| PPO with timeout bootstrap off, 5 seeds | Executed | still sags late → bootstrap not the cause |
| CleanRL PPO, 3 seeds × 500k | Executed | 3/3 hold ~490 over 400–500k |
| SAC, 3 seeds × 20k | Executed | all reach −200 by 8k; final IQM −110.8 [−111.9, −108.9] |
| mjlab install, `list-envs`, config dump (R7) | Executed | Linux, CPU |
| mjlab CPU smoke train (4 envs, 2 iters) + ONNX export | Executed | 18 steps/s; ONNX `obs[1,48] → actions[1,12]`, normaliser inside |
| Resume from checkpoint | Executed | "Loading model checkpoint … model_1.pt", continued at iteration 1 |
| CLI reward-weight override | Executed | `foot_slip.weight: 0.0` in `params/env.yaml` |
| `mjx/variants.py`: 6 variants | Executed | registered; events per variant checked; NoDR trained 1 iteration |
| `scaling_plot.py` | Executed (smoke run, no reward tag yet) | table prints |
| `export-scene` + mesh copy | Executed | scene loads in MuJoCo 3.11.0 |
| `parity_check.py` | Executed | PASS: obs ≤ 2.4e-7, one step ≤ 4.7e-3 rad, config match |
| `shift_eval.py` | Executed with an **untrained** policy | all 7 shifts run, randomized starts, height-based falls |
| mjlab env on Apple Silicon | Desk-checked | mjlab supports macOS for evaluation; install on Day 1 |
| `gate_t4.sh`, Kaggle loop, GPU training, scaling, ablations | Desk-checked | needs a T4 |
| `go1_ros_node.py` | Desk-checked | syntax-checked; RoboStack unreachable here |
| LeRobot commands | Desk-checked | against the 0.7.0 wheel's source; datasets unreachable here |
