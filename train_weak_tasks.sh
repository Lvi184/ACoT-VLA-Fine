#!/bin/bash
cd /root/gpufree-data/ACoT-VLA
source .venv/bin/activate

export DEBUG_MODE="false"
export WANDB_MODE="offline"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.4"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_PYTHON_CLIENT_MAX_FRACTIONAL_ALLOC="0.6"
export JAX_ENABLE_X64="false"
export JAX_DEFAULT_DTYPE_BITS="32"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_latency_hiding_scheduler=false"

# 关键：绕过 tyro CLI 解析
export USE_DIRECT_CONFIG="true"

# 启动训练：只在弱任务上训练 phase/memory/stage 三个新模块，从 all_improvements_v1 继续
python scripts/train.py --config-name=acot_icra_simulation_challenge_weak_tasks_phase_mem_stage --exp-name=weak_tasks_under_09_oversampled_from_v1
