"""
Stable Baselines 3 Training Utilities for MetaMachine

This module provides utilities for easy setup of SB3 training with MetaMachine environments:
- RewardComponentCallback: Logs reward component breakdowns to TensorBoard
- ProgressBarCallback: Rich progress bar with custom experiment name
- setup_sb3_training(): One-liner to configure logger and callbacks
- SB3Trainer: High-level wrapper for complete training setup

Example usage:

    # Simple one-liner setup
    from metamachine.utils.sb3_utils import setup_sb3_training
    
    model = CrossQ("MlpPolicy", env)
    callbacks = setup_sb3_training(model, env, exp_name="My Experiment")
    model.learn(total_timesteps=1000000, callback=callbacks)

    # Or use the high-level trainer
    from metamachine.utils.sb3_utils import SB3Trainer
    
    trainer = SB3Trainer(env, algorithm="CrossQ", exp_name="My Experiment")
    trainer.learn(total_timesteps=1000000)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, Union

# Lazy imports for optional SB3 dependency
if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger
    import gymnasium as gym


__all__ = [
    "RewardComponentCallback",
    "ProgressBarCallback", 
    "setup_sb3_training",
    "SB3Trainer",
    "load_from_checkpoint",
    "play_checkpoint",
]


# =============================================================================
# Callbacks
# =============================================================================

class RewardComponentCallback:
    """Callback for logging reward component values to TensorBoard/logger.
    
    This callback extracts reward component values from the environment's info dict
    and logs them to SB3's logger for visualization in TensorBoard.
    
    The reward components are expected to be in info['reward_components'] as a dict
    mapping component names to their weighted values.
    
    Example:
        callback = RewardComponentCallback()
        model.learn(total_timesteps=1000000, callback=callback)
    """
    
    def __new__(cls, verbose: int = 0):
        """Create and return a RewardComponentCallback instance."""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class _RewardComponentCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose)
                self.reward_components_sum = {}
                self.step_count = 0
            
            def _on_training_start(self) -> None:
                self.reward_components_sum = {}
                self.step_count = 0
            
            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                
                for info in infos:
                    if isinstance(info, dict) and "reward_components" in info:
                        reward_components = info["reward_components"]
                        for comp_name, comp_value in reward_components.items():
                            if comp_name not in self.reward_components_sum:
                                self.reward_components_sum[comp_name] = 0.0
                            self.reward_components_sum[comp_name] += comp_value
                
                self.step_count += 1
                return True
            
            def _on_rollout_end(self) -> None:
                if self.step_count > 0 and self.reward_components_sum:
                    for comp_name, comp_sum in self.reward_components_sum.items():
                        mean_value = comp_sum / self.step_count
                        self.logger.record(f"reward/{comp_name}", mean_value)
                
                self.reward_components_sum = {}
                self.step_count = 0
        
        return _RewardComponentCallback(verbose)


class ProgressBarCallback:
    """Rich progress bar callback with custom experiment name.
    
    Displays a beautiful progress bar using tqdm.rich with the experiment name.
    
    Example:
        callback = ProgressBarCallback(name="Tripod Training")
        model.learn(total_timesteps=1000000, callback=callback)
    """
    
    def __new__(cls, name: str = "Training"):
        """Create and return a ProgressBarCallback instance."""
        from stable_baselines3.common.callbacks import ProgressBarCallback as SB3ProgressBar
        
        try:
            from tqdm.rich import tqdm
        except ImportError:
            tqdm = None
        
        class _ProgressBarCallback(SB3ProgressBar):
            def __init__(self, name: str):
                super().__init__()
                self._name = name
                if tqdm is None:
                    raise ImportError(
                        "You must install tqdm and rich for the progress bar callback. "
                        "Install with: pip install tqdm rich"
                    )
            
            def _on_training_start(self) -> None:
                self.pbar = tqdm(
                    total=self.locals["total_timesteps"] - self.model.num_timesteps,
                    desc=f"[deep_pink1]{self._name}"
                )
        
        return _ProgressBarCallback(name)


class EpisodeStatsCallback:
    """Callback for logging additional episode statistics.
    
    Logs episode-level metrics like total reward, episode length, 
    and any custom metrics from the info dict.
    """
    
    def __new__(cls, verbose: int = 0):
        """Create and return an EpisodeStatsCallback instance."""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class _EpisodeStatsCallback(BaseCallback):
            def __init__(self, verbose: int = 0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                dones = self.locals.get("dones", [])
                
                for i, (info, done) in enumerate(zip(infos, dones)):
                    if done and isinstance(info, dict):
                        if "total_reward" in info:
                            self.episode_rewards.append(info["total_reward"])
                        if "episode_step" in info:
                            self.episode_lengths.append(info["episode_step"])
                
                return True
            
            def _on_rollout_end(self) -> None:
                if self.episode_rewards:
                    self.logger.record("episode/total_reward", 
                                      sum(self.episode_rewards) / len(self.episode_rewards))
                if self.episode_lengths:
                    self.logger.record("episode/length",
                                      sum(self.episode_lengths) / len(self.episode_lengths))
                
                self.episode_rewards = []
                self.episode_lengths = []
        
        return _EpisodeStatsCallback(verbose)


# =============================================================================
# Setup Functions
# =============================================================================

def setup_sb3_training(
    model: "BaseAlgorithm",
    env: "gym.Env",
    exp_name: str = "Training",
    log_dir: Optional[str] = None,
    checkpoint_freq: int = 100000,
    log_reward_components: bool = True,
    show_progress_bar: bool = True,
    log_episode_stats: bool = True,
    logger_outputs: List[str] = ["stdout", "csv", "tensorboard"],
    extra_callbacks: Optional[List["BaseCallback"]] = None,
) -> List["BaseCallback"]:
    """Setup SB3 training with logging and callbacks in one call.
    
    This function configures:
    - Logger with TensorBoard, CSV, and stdout outputs
    - Checkpoint callback for saving models
    - Progress bar with experiment name
    - Reward component logging
    - Episode statistics logging
    
    Args:
        model: The SB3 model to configure
        env: The environment (used to get log_dir if not specified)
        exp_name: Experiment name for progress bar and logging
        log_dir: Directory for logs. If None, uses env._log_dir
        checkpoint_freq: Save checkpoint every N steps (0 to disable)
        log_reward_components: Whether to log reward component breakdown
        show_progress_bar: Whether to show rich progress bar
        log_episode_stats: Whether to log episode statistics
        logger_outputs: List of logger outputs (stdout, csv, tensorboard, json)
        extra_callbacks: Additional callbacks to include
    
    Returns:
        List of callbacks to pass to model.learn()
    
    Example:
        model = CrossQ("MlpPolicy", env)
        callbacks = setup_sb3_training(
            model, env,
            exp_name="Tripod Locomotion",
            checkpoint_freq=50000,
        )
        model.learn(total_timesteps=1000000, callback=callbacks)
    """
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    
    # Determine log directory
    if log_dir is None:
        log_dir = getattr(env, "_log_dir", None)
        if log_dir is None:
            log_dir = f"./logs/{exp_name.replace(' ', '_').lower()}"
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = configure(log_dir, logger_outputs)
    model.set_logger(logger)
    
    # Build callbacks list
    callbacks = []
    
    # Checkpoint callback
    if checkpoint_freq > 0:
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="rl_model",
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_cb)
    
    # Progress bar callback
    if show_progress_bar:
        try:
            progress_cb = ProgressBarCallback(name=exp_name)
            callbacks.append(progress_cb)
        except ImportError:
            print("[Warning] tqdm/rich not installed, skipping progress bar")
    
    # Reward component callback
    if log_reward_components:
        reward_cb = RewardComponentCallback()
        callbacks.append(reward_cb)
    
    # Episode stats callback
    if log_episode_stats:
        stats_cb = EpisodeStatsCallback()
        callbacks.append(stats_cb)
    
    # Add extra callbacks
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    print(f"[SB3 Setup] Log directory: {log_dir}")
    print(f"[SB3 Setup] Callbacks: {[type(cb).__name__ for cb in callbacks]}")
    
    return callbacks


# =============================================================================
# High-Level Trainer Wrapper
# =============================================================================

class SB3Trainer:
    """High-level wrapper for SB3 training with MetaMachine environments.
    
    Provides a simple interface to train with any SB3-compatible algorithm
    with automatic logging, checkpointing, and callback setup.
    
    Example:
        # Basic usage
        trainer = SB3Trainer(env, algorithm="CrossQ", exp_name="Tripod Training")
        trainer.learn(total_timesteps=1000000)
        
        # With custom algorithm and policy kwargs
        trainer = SB3Trainer(
            env,
            algorithm="SAC",
            policy="MlpPolicy",
            policy_kwargs={"net_arch": [256, 256]},
            exp_name="Custom SAC Training",
        )
        trainer.learn(total_timesteps=500000)
        
        # Load and continue training
        trainer = SB3Trainer.load("./logs/my_exp/rl_model_100000_steps.zip", env)
        trainer.learn(total_timesteps=100000)  # Continue for 100k more steps
    """
    
    # Supported algorithms and their import paths
    ALGORITHMS = {
        # Stable-Baselines3 core
        "PPO": ("stable_baselines3", "PPO"),
        "SAC": ("stable_baselines3", "SAC"),
        "TD3": ("stable_baselines3", "TD3"),
        "A2C": ("stable_baselines3", "A2C"),
        "DDPG": ("stable_baselines3", "DDPG"),
        "DQN": ("stable_baselines3", "DQN"),
        # SB3-Contrib
        "CrossQ": ("sb3_contrib", "CrossQ"),
        "TQC": ("sb3_contrib", "TQC"),
        "TRPO": ("sb3_contrib", "TRPO"),
        "ARS": ("sb3_contrib", "ARS"),
        "RecurrentPPO": ("sb3_contrib", "RecurrentPPO"),
    }
    
    def __init__(
        self,
        env: "gym.Env",
        algorithm: Union[str, Type["BaseAlgorithm"]] = "CrossQ",
        policy: str = "MlpPolicy",
        exp_name: str = "SB3 Training",
        log_dir: Optional[str] = None,
        checkpoint_freq: int = 100000,
        seed: Optional[int] = None,
        device: str = "auto",
        verbose: int = 1,
        policy_kwargs: Optional[dict] = None,
        algorithm_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the SB3 trainer.
        
        Args:
            env: The gymnasium environment
            algorithm: Algorithm name (e.g., "CrossQ", "SAC", "PPO") or class
            policy: Policy type (e.g., "MlpPolicy", "CnnPolicy")
            exp_name: Experiment name for logging
            log_dir: Directory for logs (uses env._log_dir if None)
            checkpoint_freq: Save checkpoint every N steps (0 to disable)
            seed: Random seed
            device: Device to use ("auto", "cuda", "cpu")
            verbose: Verbosity level
            policy_kwargs: Additional kwargs for policy network
            algorithm_kwargs: Additional kwargs for algorithm
            **kwargs: Additional kwargs passed to algorithm
        """
        self.env = env
        self.exp_name = exp_name
        self.checkpoint_freq = checkpoint_freq
        
        # Determine log directory
        if log_dir is None:
            self.log_dir = getattr(env, "_log_dir", None)
            if self.log_dir is None:
                self.log_dir = f"./logs/{exp_name.replace(' ', '_').lower()}"
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Get algorithm class
        if isinstance(algorithm, str):
            algorithm_cls = self._get_algorithm_class(algorithm)
        else:
            algorithm_cls = algorithm
        
        # Merge kwargs
        all_kwargs = {
            "policy": policy,
            "env": env,
            "device": device,
            "verbose": verbose,
            "seed": seed,
        }
        if policy_kwargs:
            all_kwargs["policy_kwargs"] = policy_kwargs
        if algorithm_kwargs:
            all_kwargs.update(algorithm_kwargs)
        all_kwargs.update(kwargs)
        
        # Remove None values
        all_kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
        
        # Create model
        self.model = algorithm_cls(**all_kwargs)
        
        # Setup training (logger + callbacks)
        self.callbacks = setup_sb3_training(
            self.model,
            env,
            exp_name=exp_name,
            log_dir=self.log_dir,
            checkpoint_freq=checkpoint_freq,
        )
        
        print(f"[SB3Trainer] Algorithm: {algorithm_cls.__name__}")
        print(f"[SB3Trainer] Policy: {policy}")
        print(f"[SB3Trainer] Log directory: {self.log_dir}")
    
    def _get_algorithm_class(self, name: str) -> Type["BaseAlgorithm"]:
        """Get algorithm class by name."""
        if name not in self.ALGORITHMS:
            available = ", ".join(self.ALGORITHMS.keys())
            raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")
        
        module_name, class_name = self.ALGORITHMS[name]
        
        try:
            import importlib
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Could not import {class_name} from {module_name}. "
                f"Install with: pip install {module_name.replace('_', '-')}"
            ) from e
    
    def learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,  # We use our own
        **kwargs,
    ) -> "SB3Trainer":
        """Train the model.
        
        Args:
            total_timesteps: Total number of timesteps to train
            reset_num_timesteps: Whether to reset timestep counter
            **kwargs: Additional kwargs passed to model.learn()
        
        Returns:
            self for chaining
        """
        print(f"\n[SB3Trainer] Starting training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            **kwargs,
        )
        
        print(f"\n[SB3Trainer] Training complete!")
        print(f"[SB3Trainer] Logs saved to: {self.log_dir}")
        
        return self
    
    def save(self, path: Optional[str] = None) -> str:
        """Save the model.
        
        Args:
            path: Path to save to. If None, saves to log_dir/final_model.zip
        
        Returns:
            Path where model was saved
        """
        if path is None:
            path = os.path.join(self.log_dir, "final_model")
        
        self.model.save(path)
        print(f"[SB3Trainer] Model saved to: {path}")
        return path
    
    @classmethod
    def load(
        cls,
        path: str,
        env: "gym.Env",
        exp_name: str = "Continued Training",
        **kwargs,
    ) -> "SB3Trainer":
        """Load a model and create a trainer for continued training.
        
        Args:
            path: Path to the saved model
            env: Environment to use
            exp_name: Experiment name for new logs
            **kwargs: Additional kwargs
        
        Returns:
            SB3Trainer instance with loaded model
        """
        # Detect algorithm from file (simple heuristic)
        # In practice, you might want to save metadata with the model
        from stable_baselines3.common.base_class import BaseAlgorithm
        
        # Try to load with different algorithms
        model = None
        for algo_name, (module_name, class_name) in cls.ALGORITHMS.items():
            try:
                import importlib
                module = importlib.import_module(module_name)
                algo_cls = getattr(module, class_name)
                model = algo_cls.load(path, env=env)
                print(f"[SB3Trainer] Loaded model with algorithm: {algo_name}")
                break
            except Exception:
                continue
        
        if model is None:
            raise ValueError(f"Could not load model from {path}")
        
        # Create trainer wrapper
        trainer = cls.__new__(cls)
        trainer.env = env
        trainer.exp_name = exp_name
        trainer.model = model
        trainer.log_dir = getattr(env, "_log_dir", f"./logs/{exp_name.replace(' ', '_').lower()}")
        trainer.checkpoint_freq = kwargs.get("checkpoint_freq", 100000)
        
        # Setup callbacks for continued training
        trainer.callbacks = setup_sb3_training(
            model,
            env,
            exp_name=exp_name,
            log_dir=trainer.log_dir,
            checkpoint_freq=trainer.checkpoint_freq,
        )
        
        return trainer
    
    def add_callback(self, callback: "BaseCallback") -> "SB3Trainer":
        """Add an additional callback.
        
        Args:
            callback: Callback to add
        
        Returns:
            self for chaining
        """
        self.callbacks.append(callback)
        return self
    
    @property
    def num_timesteps(self) -> int:
        """Get the current number of timesteps."""
        return self.model.num_timesteps


# =============================================================================
# Checkpoint Loading Utilities
# =============================================================================

def load_from_checkpoint(
    log_dir: str,
    checkpoint: Optional[str] = None,
    render_mode: str = "viewer",
    real_robot: bool = False,
    device: str = "auto",
) -> tuple:
    """Load environment and model from a training checkpoint directory.
    
    This function recreates the exact environment used during training by loading
    the saved config.yaml from the log directory, and optionally loads a model
    checkpoint.
    
    Args:
        log_dir: Path to the training log directory (contains config.yaml)
        checkpoint: Model checkpoint to load. Can be:
            - None: Find the latest checkpoint automatically
            - "latest": Find the latest checkpoint automatically
            - "best": Find the best checkpoint (if exists)
            - "final": Load final_model.zip
            - Path to specific checkpoint file
            - Integer: Load rl_model_{checkpoint}_steps.zip
        render_mode: Render mode for simulation ("viewer", "mp4", "none")
        real_robot: If True, create RealMetaMachine instead of MetaMachine
        device: Device for model loading ("auto", "cuda", "cpu")
    
    Returns:
        tuple: (env, model, cfg) - environment, loaded model (or None), config
    
    Example:
        # Load latest checkpoint with viewer
        env, model, cfg = load_from_checkpoint("./logs/my_experiment")
        
        # Load specific checkpoint
        env, model, cfg = load_from_checkpoint(
            "./logs/my_experiment",
            checkpoint=500000,
            render_mode="viewer"
        )
        
        # Load for real robot deployment
        env, model, cfg = load_from_checkpoint(
            "./logs/my_experiment",
            checkpoint="final",
            real_robot=True
        )
    """
    from pathlib import Path
    import glob
    
    log_path = Path(log_dir)
    
    # Validate log directory
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    config_path = log_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"[Checkpoint] Loading from: {log_dir}")
    
    # Load configuration
    from metamachine.environments.configs.config_registry import ConfigRegistry
    cfg = ConfigRegistry.create_from_file(str(config_path))
    
    # Override render mode for simulation
    if not real_robot:
        cfg.simulation.render_mode = render_mode
        cfg.simulation.render = render_mode != "none"
        # Disable video recording during playback
        cfg.simulation.video_record_interval = 0
    
    # Create environment
    if real_robot:
        from metamachine.environments.env_real import RealMetaMachine
        cfg.environment.mode = "real"
        env = RealMetaMachine(cfg)
        print(f"[Checkpoint] Created RealMetaMachine environment")
    else:
        from metamachine.environments.env_sim import MetaMachine
        cfg.environment.mode = "sim"
        env = MetaMachine(cfg)
        print(f"[Checkpoint] Created MetaMachine environment (render_mode={render_mode})")
    
    # Find checkpoint file
    checkpoint_path = _resolve_checkpoint_path(log_path, checkpoint)
    
    # Load model if checkpoint found
    model = None
    if checkpoint_path is not None:
        model = _load_sb3_model(checkpoint_path, env, device)
    else:
        print(f"[Checkpoint] No checkpoint found, returning environment only")
    
    return env, model, cfg


def _resolve_checkpoint_path(log_path: Path, checkpoint: Optional[str]) -> Optional[str]:
    """Resolve checkpoint specification to actual file path."""
    import glob
    
    if checkpoint is None or checkpoint == "latest":
        # Find latest checkpoint by step number
        pattern = str(log_path / "rl_model_*_steps.zip")
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            # Try final_model
            final_path = log_path / "final_model.zip"
            if final_path.exists():
                return str(final_path)
            return None
        
        # Sort by step number and get latest
        def get_steps(path):
            try:
                name = Path(path).stem
                return int(name.replace("rl_model_", "").replace("_steps", ""))
            except:
                return 0
        
        checkpoints.sort(key=get_steps, reverse=True)
        checkpoint_path = checkpoints[0]
        print(f"[Checkpoint] Found latest: {Path(checkpoint_path).name}")
        return checkpoint_path
    
    elif checkpoint == "final":
        final_path = log_path / "final_model.zip"
        if final_path.exists():
            print(f"[Checkpoint] Loading final model")
            return str(final_path)
        raise FileNotFoundError(f"Final model not found: {final_path}")
    
    elif checkpoint == "best":
        best_path = log_path / "best_model.zip"
        if best_path.exists():
            print(f"[Checkpoint] Loading best model")
            return str(best_path)
        raise FileNotFoundError(f"Best model not found: {best_path}")
    
    elif isinstance(checkpoint, int) or checkpoint.isdigit():
        # Load specific step checkpoint
        steps = int(checkpoint)
        checkpoint_path = log_path / f"rl_model_{steps}_steps.zip"
        if checkpoint_path.exists():
            print(f"[Checkpoint] Loading step {steps}")
            return str(checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    else:
        # Assume it's a direct path
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = log_path / checkpoint
        if checkpoint_path.exists():
            print(f"[Checkpoint] Loading: {checkpoint_path.name}")
            return str(checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def _load_sb3_model(checkpoint_path: str, env, device: str = "auto"):
    """Load an SB3 model from checkpoint, auto-detecting algorithm."""
    
    # Try loading with different algorithms
    for algo_name, (module_name, class_name) in SB3Trainer.ALGORITHMS.items():
        try:
            import importlib
            module = importlib.import_module(module_name)
            algo_cls = getattr(module, class_name)
            model = algo_cls.load(checkpoint_path, env=env, device=device)
            print(f"[Checkpoint] Loaded model (algorithm: {algo_name})")
            return model
        except Exception:
            continue
    
    raise ValueError(f"Could not load model from {checkpoint_path}. "
                    "Tried all supported algorithms.")


def play_checkpoint(
    log_dir: str,
    checkpoint: Optional[str] = None,
    num_episodes: int = 5,
    render_mode: str = "viewer",
    real_robot: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
    commands: Optional[dict] = None,
    disable_resampling: bool = False,
) -> dict:
    """Play/evaluate a trained policy from a checkpoint.
    
    Convenience function that loads a checkpoint and runs episodes, displaying
    the robot behavior. Useful for quick visualization and evaluation.
    
    Args:
        log_dir: Path to the training log directory
        checkpoint: Checkpoint to load (see load_from_checkpoint for options)
        num_episodes: Number of episodes to run (0 = run forever)
        render_mode: Render mode ("viewer", "mp4", "none")
        real_robot: If True, deploy to real robot
        deterministic: If True, use deterministic policy (no exploration noise)
        verbose: If True, print episode statistics
        commands: Optional dict of command values to set (e.g., {"turn_rate": 1.5}).
            If provided, these commands will be set after each reset.
            Use with disable_resampling=True to keep commands fixed.
        disable_resampling: If True, disable automatic command resampling.
            Useful when you want to test specific command values.
    
    Returns:
        dict: Statistics from the playback (rewards, lengths, etc.)
    
    Example:
        # Quick visualization
        play_checkpoint("./logs/my_experiment")
        
        # Evaluate for 10 episodes
        stats = play_checkpoint("./logs/my_experiment", num_episodes=10)
        print(f"Mean reward: {stats['mean_reward']:.2f}")
        
        # Deploy to real robot
        play_checkpoint("./logs/my_experiment", real_robot=True)
        
        # Test specific behavior (e.g., turn left)
        play_checkpoint(
            "./logs/my_experiment",
            commands={"turn_rate": 1.5},      # Turn left
            disable_resampling=True,           # Keep command fixed
        )
        
        # Test going straight
        play_checkpoint(
            "./logs/my_experiment",
            commands={"turn_rate": 0.0, "forward_speed": 0.5},
            disable_resampling=True,
        )
        
        # Test one-hot commands by index (for onehot_turning config)
        # 0=straight, 1=left, 2=right
        play_checkpoint(
            "./logs/my_experiment",
            commands={"_onehot_index": 1},    # Turn left mode
            disable_resampling=True,
        )
        
        # Test one-hot commands by name
        play_checkpoint(
            "./logs/my_experiment",
            commands={"_onehot_name": "cmd_right"},  # Turn right mode
            disable_resampling=True,
        )
    """
    import numpy as np
    
    # Load environment and model
    env, model, cfg = load_from_checkpoint(
        log_dir,
        checkpoint=checkpoint,
        render_mode=render_mode,
        real_robot=real_robot,
    )
    
    if model is None:
        raise ValueError("No model checkpoint found to play")
    
    # Disable command resampling if requested
    if disable_resampling:
        _disable_command_resampling(env)
    
    print(f"\n{'=' * 60}")
    print(f"Playing Policy")
    print(f"{'=' * 60}")
    print(f"  Episodes: {'infinite' if num_episodes == 0 else num_episodes}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Real robot: {real_robot}")
    if commands:
        print(f"  Commands: {commands}")
    if disable_resampling:
        print(f"  Command resampling: DISABLED")
    print(f"{'=' * 60}\n")
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    episode_count = 0
    
    try:
        while num_episodes == 0 or episode_count < num_episodes:
            obs, info = env.reset()
            
            # Set commands after reset if specified
            if commands:
                _set_commands(env, commands, verbose=verbose and episode_count == 0)
                # Get updated observation with new commands
                obs = _get_observation_with_commands(env)
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            if verbose:
                print(f"Episode {episode_count}: Reward = {episode_reward:.2f}, "
                      f"Length = {episode_length}")
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        env.close()
    
    # Compute statistics
    stats = {
        "num_episodes": len(episode_rewards),
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
        "std_reward": np.std(episode_rewards) if episode_rewards else 0,
        "min_reward": np.min(episode_rewards) if episode_rewards else 0,
        "max_reward": np.max(episode_rewards) if episode_rewards else 0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "commands": commands,
    }
    
    if verbose and episode_rewards:
        print(f"\n{'=' * 60}")
        print(f"Summary ({len(episode_rewards)} episodes)")
        print(f"{'=' * 60}")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
        if commands:
            print(f"  Commands used: {commands}")
        print(f"{'=' * 60}")
    
    return stats


def _set_commands(env, commands: dict, verbose: bool = False) -> None:
    """Set command values on the environment.
    
    Args:
        env: The environment with a state and command_manager
        commands: Dict mapping command names to values, or special keys:
            - "_onehot_index": int - Set one-hot by index (0, 1, 2, ...)
            - "_onehot_name": str - Set one-hot by name (e.g., "cmd_left")
            - Regular keys: Set individual command values
        verbose: If True, print the commands being set
    
    Example:
        # Set individual commands
        _set_commands(env, {"turn_rate": 1.5, "forward_speed": 0.5})
        
        # Set one-hot by index (0=straight, 1=left, 2=right)
        _set_commands(env, {"_onehot_index": 1})  # Turn left
        
        # Set one-hot by name
        _set_commands(env, {"_onehot_name": "cmd_left"})
    """
    # Try to access command manager through different paths
    command_manager = None
    
    # Path 1: env.state.command_manager
    if hasattr(env, 'state') and hasattr(env.state, 'command_manager'):
        command_manager = env.state.command_manager
    # Path 2: env._command_manager
    elif hasattr(env, '_command_manager'):
        command_manager = env._command_manager
    # Path 3: env.command_manager
    elif hasattr(env, 'command_manager'):
        command_manager = env.command_manager
    
    if command_manager is None:
        if verbose:
            print("[Warning] Could not find command manager, commands not set")
        return
    
    # Handle special one-hot commands
    if "_onehot_index" in commands:
        idx = int(commands["_onehot_index"])
        if hasattr(command_manager, 'set_onehot_by_index'):
            command_manager.set_onehot_by_index(idx)
            if verbose:
                names = getattr(command_manager, 'command_names', [])
                name = names[idx] if idx < len(names) else f"index_{idx}"
                print(f"  Set one-hot command: index={idx} ({name})")
        else:
            # Fallback: manually set one-hot
            for i in range(command_manager.num_commands):
                command_manager.set_command(i, 1.0 if i == idx else 0.0)
            if verbose:
                print(f"  Set one-hot command: index={idx}")
        return
    
    if "_onehot_name" in commands:
        name = commands["_onehot_name"]
        if hasattr(command_manager, 'set_onehot_by_name'):
            command_manager.set_onehot_by_name(name)
            if verbose:
                print(f"  Set one-hot command: {name}")
        else:
            # Fallback: find index and set manually
            try:
                idx = command_manager.command_names.index(name)
                for i in range(command_manager.num_commands):
                    command_manager.set_command(i, 1.0 if i == idx else 0.0)
                if verbose:
                    print(f"  Set one-hot command: {name}")
            except ValueError:
                if verbose:
                    print(f"  [Warning] Command '{name}' not found")
        return
    
    # Set each command by name (regular mode)
    for cmd_name, cmd_value in commands.items():
        try:
            command_manager.set_command_by_name(cmd_name, cmd_value)
            if verbose:
                print(f"  Set command '{cmd_name}' = {cmd_value}")
        except (ValueError, KeyError) as e:
            if verbose:
                print(f"  [Warning] Could not set command '{cmd_name}': {e}")


def _disable_command_resampling(env) -> None:
    """Disable automatic command resampling on the environment.
    
    Args:
        env: The environment with a command_manager
    """
    # Try to access command manager through different paths
    command_manager = None
    
    if hasattr(env, 'state') and hasattr(env.state, 'command_manager'):
        command_manager = env.state.command_manager
    elif hasattr(env, '_command_manager'):
        command_manager = env._command_manager
    elif hasattr(env, 'command_manager'):
        command_manager = env.command_manager
    
    if command_manager is not None:
        # Set resampling interval to 0 or very large number to disable
        command_manager.resampling_interval = 0


def _get_observation_with_commands(env):
    """Get observation from environment after commands have been updated.
    
    This is needed because the observation may include command values,
    and we need to refresh it after setting new commands.
    
    Args:
        env: The environment
        
    Returns:
        Updated observation array
    """
    # Try to get fresh observation from state
    if hasattr(env, 'state') and hasattr(env.state, 'get_observation'):
        return env.state.get_observation(insert=False, reset=False)
    
    # Fallback: return observation space sample (not ideal but safe)
    # The next step will get the correct observation anyway
    if hasattr(env, '_last_obs'):
        return env._last_obs
    
    return env.observation_space.sample()

