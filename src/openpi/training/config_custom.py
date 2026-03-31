from openpi.training.config import *

@dataclass
class CustomICRAConfig(TrainingConfig):
    name = "custom_icra_open_door"
    
    model = acot_vla.ACOTConfig(
        coarse_action_horizon=30,
        action_horizon=30,
        paligemma_variant="gemma_2b_lora",
        adopt_explicit_action_reasoner=True,
        adopt_implicit_action_reasoner=True,
        downsample_based_implicit_extractor=True
    )
    
    data = LerobotACOTGo2DataConfig(
        default_prompt="Open the door task",
        repo_id=[
            "/root/gpufree-data/AgiBotWorldChallenge-2026/agibot_data_without_depth/open_door",],
        assets=AssetsConfig(
            assets_dir=None,
            asset_id="/root/gpufree-data/ACoT-VLA/assets/open_door",
        ),
        prompt_map_inject_to_training={
            "Turn the doorknob": ("Turn the doorknob and push the door", 0.5),
        },
    )
    
    training = TrainingHyperparametersConfig(
        batch_size=8,
        num_workers=4,
        learning_rate=1e-4,
        num_epochs=100,
    )

# 注册配置
TRAINING_CONFIGS["custom_icra_open_door"] = CustomICRAConfig()
