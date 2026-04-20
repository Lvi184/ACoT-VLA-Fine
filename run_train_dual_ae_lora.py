#!/usr/bin/env python3
"""
Simple training script that bypasses tyro CLI parsing issues.
Directly loads the config and runs training.
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

import dataclasses
import logging
import platform
from typing import Any
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Import the train script functions
sys.path.insert(0, '/root/gpufree-data/ACoT-VLA/scripts')

# Now let's directly use the existing config but modify it
print("Loading base config...")
base_config = _config.get_config('acot_icra_simulation_challenge_reasoning_to_action')

print("Creating modified config with Dual AE LoRA...")
# Create a new config with Dual AE LoRA
modified_config = dataclasses.replace(
    base_config,
    name="acot_icra_simulation_challenge_dual_ae_lora",
    model=dataclasses.replace(
        base_config.model,
        coarse_action_expert_variant="gemma_300m_lora",
        action_expert_variant="gemma_300m_lora",
    ),
)

# Update the exp_name
modified_config = dataclasses.replace(
    modified_config,
    exp_name="finetune_dual_ae_lora_v1",
)

print(f"Config loaded successfully!")
print(f"  paligemma_variant: {modified_config.model.paligemma_variant}")
print(f"  coarse_action_expert_variant: {modified_config.model.coarse_action_expert_variant}")
print(f"  action_expert_variant: {modified_config.model.action_expert_variant}")
print(f"  exp_name: {modified_config.exp_name}")

# Now we need to run the training. Let's import the main function from train.py
print("\nStarting training...")
print("Note: This is a simplified script. For full functionality, please fix the tyro CLI issue.")
print("Alternatively, use the original config and manually modify the model in the code.")

print("\nFor now, let's just verify the config works by attempting to create the model...")

# Test creating the model
try:
    rng = jax.random.PRNGKey(42)
    model = modified_config.model.create(rng)
    print("✅ Model created successfully!")
    
    # Count parameters
    params = nnx.state(model)
    num_params = training_utils.count_parameters(params)
    print(f"✅ Total parameters: {num_params:,}")
    
    print("\nConfig is valid! To run training, please:")
    print("1. Fix the tyro CLI issue in config.py")
    print("2. Or modify train.py to directly accept a config object")
    print("3. Or use the original config and manually set Dual AE variants to gemma_300m_lora")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
