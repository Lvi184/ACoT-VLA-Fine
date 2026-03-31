#!/bin/bash
# Training script for acot_icra_finetune_low_score_tasks_frozen_lora

cd /root/gpufree-data/ACoT-VLA

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export DEBUG_MODE="false"

# Run training
python scripts/train.py \
  --config=acot_icra_finetune_low_score_tasks_frozen_lora \
  --exp_name=finetune_low_score_tasks_frozen_lora_v1 \
  --overwrite=False \
  --resume=False
