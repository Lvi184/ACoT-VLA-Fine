#!/usr/bin/env python3
"""
Direct training script that bypasses tyro CLI parsing.
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, '/root/gpufree-data/ACoT-VLA')

# Set environment variables first
os.environ["DEBUG_MODE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MAX_FRACTIONAL_ALLOC"] = "0.6"
os.environ["JAX_ENABLE_X64"] = "false"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_latency_hiding_scheduler=false"

# Import the necessary modules
import openpi.training.config as _config

# Get our modified config directly
print("Loading config...")
config = _config.get_config('acot_icra_simulation_challenge_reasoning_to_action')

# Override exp_name
config = config.replace(exp_name="finetune_dual_ae_lora_v1")

print(f"Config loaded successfully!")
print(f"  paligemma_variant: {config.model.paligemma_variant}")
print(f"  coarse_action_expert_variant: {config.model.coarse_action_expert_variant}")
print(f"  action_expert_variant: {config.model.action_expert_variant}")
print(f"  exp_name: {config.exp_name}")

# Now import and run the main function from train.py
print("\nImporting train module...")
sys.path.insert(0, '/root/gpufree-data/ACoT-VLA/scripts')

# We need to import after setting up the environment first
print("Starting training...")

# Let's try to import the main function
from train import main

# Run it!
print("Calling main()...")
main(config)
