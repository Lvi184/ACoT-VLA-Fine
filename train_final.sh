#!/bin/bash
# Final training script - uses your original successful script
# Config file already modified: Dual AE now uses gemma_300m_lora

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

# Run training with modified config (Dual AE now uses gemma_300m_lora)
python scripts/train.py \
  --config=acot_icra_simulation_challenge_reasoning_to_action \
  --exp-name=finetune_dual_ae_lora_v1 \
  --overwrite=False \
  --resume=False
