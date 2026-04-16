
#!/bin/bash
cd /root/gpufree-data/ACoT-VLA
source .venv/bin/activate

export DEBUG_MODE="false"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.4"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
export XLA_PYTHON_CLIENT_MAX_FRACTIONAL_ALLOC="0.6"
export JAX_ENABLE_X64="false"
export JAX_DEFAULT_DTYPE_BITS="32"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_latency_hiding_scheduler=false"

# 关键：绕过 tyro CLI 解析
export USE_DIRECT_CONFIG="true"

python scripts/train.py --config-name=acot_icra_simulation_challenge_all_improvements --exp-name=all_improvements_v2
