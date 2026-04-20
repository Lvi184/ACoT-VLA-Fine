#!/bin/bash
# 训练脚本 - 方案 B：部分解冻官方 LoRA
# PaliGemma LoRA 和 Dual AE LoRA 都可以训练

cd /root/gpufree-data/ACoT-VLA

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export DEBUG_MODE="false"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.4"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_PYTHON_CLIENT_MAX_FRACTIONAL_ALLOC="0.6"
export JAX_ENABLE_X64="false"
export JAX_DEFAULT_DTYPE_BITS="32"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_latency_hiding_scheduler=false"

# Enable direct config loading (bypass tyro CLI)
export USE_DIRECT_CONFIG="true"
export DIRECT_CONFIG_NAME="acot_icra_simulation_challenge_reasoning_to_action_partial_unfreeze"

# Run training with PARTIAL UNFREEZE - PaliGemma LoRA also trainable
python scripts/train.py \
  --exp-name=finetune_partial_unfreeze_v1 \
  2>&1 | tee train_partial_unfreeze.log
