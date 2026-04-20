#!/usr/bin/env python3
"""
合并增量微调 checkpoint 生成完整模型 (openpi 版本)
基于:
- 基础模型: all_improvements_v1 (包含完整主干：LLM + 视觉 backbone)
- 增量微调: weak_tasks_under_09_oversampled_from_v1/14999 (只包含更新的参数)

输出: 合并后的完整模型，可以直接加载测试
"""

import os
import sys
import pathlib
import logging

import jax.numpy as jnp

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints

def main():
    # 配置信息
    config_name = "acot_icra_simulation_challenge_weak_tasks_phase_mem_stage"
    base_ckpt_dir = pathlib.Path("/root/gpufree-data/ACoT-VLA/checkpoints/acot_icra_simulation_challenge_all_improvements/all_improvements_v1/9999")
    delta_ckpt_dir = pathlib.Path("/root/gpufree-data/ACoT-VLA/checkpoints/acot_icra_simulation_challenge_weak_tasks_phase_mem_stage/weak_tasks_under_09_oversampled_from_v1/14999")
    output_ckpt_dir = pathlib.Path("/root/gpufree-data/ACoT-VLA/checkpoints/acot_icra_simulation_challenge_weak_tasks_phase_mem_stage/weak_tasks_under_09_oversampled_from_v1/merged_full_14999")
    
    print(f"Loading config: {config_name}")
    config = _config.get_config(config_name)
    
    print(f"Creating model from base checkpoint...")
    # 第一步：先从基础 checkpoint 创建完整模型
    model = config.model.load(_model.restore_params(base_ckpt_dir / "params", dtype=jnp.bfloat16))
    
    # 第二步：加载增量权重，会自动覆盖更新过的参数
    print(f"Loading delta parameters from: {delta_ckpt_dir}")
    delta_params = _model.restore_params(delta_ckpt_dir / "params", dtype=config.model.compute_dtype)
    # 再次加载，会覆盖已有的参数
    model = config.model.load(delta_params, model=model)
    
    print(f"Merged successfully, saving full checkpoint...")
    
    # 第三步：保存完整合并后的模型
    os.makedirs(output_ckpt_dir, exist_ok=True)
    # 保存参数
    _checkpoints.save_checkpoint(model, output_ckpt_dir, step=14999)
    
    # 复制 assets (norm stats 等) 从增量目录（如果有），否则从基础目录
    if os.path.exists(delta_ckpt_dir / "assets"):
        import shutil
        shutil.copytree(delta_ckpt_dir / "assets", output_ckpt_dir / "assets", dirs_exist_ok=True)
        print(f"Copied assets from delta checkpoint")
    elif os.path.exists(base_ckpt_dir / "assets"):
        import shutil
        shutil.copytree(base_ckpt_dir / "assets", output_ckpt_dir / "assets", dirs_exist_ok=True)
        print(f"Copied assets from base checkpoint")
    
    print("✅ Merge completed!")
    print("=" * 60)
    print(f"Full merged checkpoint saved at: {output_ckpt_dir}")
    print(f"You can now load this complete checkpoint directly with:")
    print(f"  python scripts/serve_policy.py --env=G2SIM policy:checkpoint --policy.config={config_name} --policy.dir={output_ckpt_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
