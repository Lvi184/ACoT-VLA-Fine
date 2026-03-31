import jax
import numpy as np
import openpi.models.model as _model

# Load the official baseline checkpoint
checkpoint_path = "/root/gpufree-data/ACoT-VLA/checkpoints/baseline/30000/params"
print(f"Loading checkpoint from: {checkpoint_path}")

# Restore the params
params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)

# Flatten the params to see all keys
import flax.traverse_util
flat_params = flax.traverse_util.flatten_dict(params, sep="/")

print("\n=== All parameter keys ===")
for key in sorted(flat_params.keys()):
    print(f"{key}: {flat_params[key].shape}")

print("\n=== LoRA-related keys ===")
lora_keys = [k for k in flat_params.keys() if "lora" in k.lower()]
for key in sorted(lora_keys):
    print(f"{key}: {flat_params[key].shape}")

print(f"\nTotal number of parameters: {len(flat_params)}")
print(f"Number of LoRA parameters: {len(lora_keys)}")
