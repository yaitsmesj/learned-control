#!/usr/bin/env bash
# Day-1 gate: can a free Kaggle T4 run mjlab Go1 at the env counts the plan needs?
# Run in a Kaggle notebook (GPU T4, Internet ON), after cloning the repo:
#   !git clone https://github.com/<you>/learned-control && cd learned-control && bash mjx/gate_t4.sh
# Out: gate/*.log, gate/gpu_mem.csv  -> fill the gate table in Reference R6.
# [temporary stand-in] Kaggle-specific; on any other CUDA box the same train lines work.
set -euo pipefail
mkdir -p gate
pip install -q "mjlab==1.6.1" onnxruntime tensorboard
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee gate/torch.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee gate/gpu.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 5 > gate/gpu_mem.csv &
MON=$!
for N in 1024 4096; do
  timeout 900 train Mjlab-Velocity-Flat-Unitree-Go1 \
    --env.scene.num-envs "$N" --agent.max-iterations 50 \
    --agent.logger tensorboard --agent.run-name "gate_$N" 2>&1 | tee "gate/train_$N.log" || true
done
kill $MON
grep -h "Steps per second" gate/train_*.log | tail -4
sort -t, -k2 -n gate/gpu_mem.csv | tail -1
