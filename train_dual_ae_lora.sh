#!/bin/bash
# Training script for acot_icra_simulation_challenge_dual_ae_lora
# Freezes PaliGemma LoRA, only trains Dual AE LoRA

cd /root/gpufree-data/ACoT-VLA

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export DEBUG_MODE="false"

# Run training
python scripts/train.py \
  --config=acot_icra_simulation_challenge_dual_ae_lora \
  --exp_name=finetune_dual_ae_lora_v1 \
  --overwrite=False \
  --resume=False
