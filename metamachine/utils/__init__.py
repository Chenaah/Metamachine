"""Utilities for MetaMachine."""

from .checkpoint_manager import (
    CheckpointManager,
    get_checkpoint,
    download_from_url,
    register_model,
    list_models,
    print_models,
    get_default_manager,
)

from .rollout_recorder import (
    RolloutRecorder,
    EpisodeData,
    StateSnapshot,
)

from .mujoco_utils import (
    find_parent_torso,
    get_all_weld_clusters,
    get_largest_weld_cluster_average_pos,
    get_weld_cluster_center_of_mass,
)

__all__ = [
    # Checkpoint manager
    "CheckpointManager",
    "get_checkpoint",
    "download_from_url",
    "register_model",
    "list_models",
    "print_models",
    "get_default_manager",
    # Rollout recorder
    "RolloutRecorder",
    "EpisodeData",
    "StateSnapshot",
    # MuJoCo utilities
    "find_parent_torso",
    "get_all_weld_clusters",
    "get_largest_weld_cluster_average_pos",
    "get_weld_cluster_center_of_mass",
    # SB3 utilities (optional, requires stable-baselines3)
    "SB3Trainer",
    "setup_sb3_training",
    "RewardComponentCallback",
    "ProgressBarCallback",
    "load_from_checkpoint",
    "play_checkpoint",
]

# SB3 utilities (optional import - only available if stable-baselines3 is installed)
try:
    from .sb3_utils import (
        SB3Trainer,
        setup_sb3_training,
        RewardComponentCallback,
        ProgressBarCallback,
        load_from_checkpoint,
        play_checkpoint,
    )
except ImportError:
    # SB3 not installed, provide placeholder
    SB3Trainer = None
    setup_sb3_training = None
    RewardComponentCallback = None
    ProgressBarCallback = None
    load_from_checkpoint = None
    play_checkpoint = None
