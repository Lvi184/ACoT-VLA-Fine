#!/bin/bash

# 方案 C：步骤连贯性损失训练脚本
# 增加步骤连贯性损失，改善多步骤任务表现

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
export DIRECT_CONFIG_NAME="acot_icra_simulation_challenge_reasoning_to_action_step_continuity"

# 运行训练
echo "=========================================="
echo "开始训练：方案 C - 步骤连贯性损失"
echo "配置名称：acot_icra_simulation_challenge_reasoning_to_action_step_continuity"
echo "=========================================="
echo ""
echo "关键特性："
echo "  - 启用步骤连贯性损失 (use_step_continuity_loss=True)"
echo "  - 损失权重：0.1 (step_continuity_loss_weight=0.1)"
echo "  - 损失类型：L2 (step_continuity_loss_type=\"l2\")"
echo "  - 冻结 PaliGemma LoRA，只训练 Dual AE LoRA"
echo ""
echo "预期效果："
echo "  - 改善多步骤任务的步骤衔接"
echo "  - 减少步骤断裂问题"
echo "  - 提升 Task_5、Task_8、Task_9 等任务得分"
echo ""

python scripts/train.py \
  --exp-name=finetune_step_continuity_v1 \
  2>&1 | tee train_step_continuity.log

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "检查点位置："
echo "  checkpoints/acot_icra_simulation_challenge_reasoning_to_action_step_continuity/finetune_step_continuity_v1/"
echo ""
echo "日志文件："
echo "  train_step_continuity.log"
