"""
Multi-Model Runner Utilities for MetaMachine

This module provides utilities for loading and managing multiple trained models
for comparison, A/B testing, or runtime model switching.

Features:
- MultiModelRunner: Manages multiple loaded SB3 models with easy switching
- load_multiple_models: Load multiple models from different log directories
- Automatic observation adaptation for models with different observation dimensions
- Support for all SB3-compatible algorithms (CrossQ, SAC, PPO, etc.)

Example usage:

    from metamachine.utils.multi_model import load_multiple_models

    # Load multiple trained models
    runner, first_cfg = load_multiple_models([
        "logs/experiment1",
        "logs/experiment2",
        "logs/experiment3",
    ])

    # Use in real robot control
    env = RealMetaMachine(first_cfg)
    env.model_runner = runner

    # Get predictions with automatic model switching
    action, _ = runner.predict(observation)

    # Switch models
    runner.next_model()  # or runner.prev_model()

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations

import glob
import json
import os
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


__all__ = [
    "MultiModelRunner",
    "load_multiple_models",
    "load_model_standalone",
    "find_checkpoint_path",
]


class MultiModelRunner:
    """Manages multiple loaded models for easy switching during runtime.
    
    This class enables:
    - Loading multiple SB3 models from different training runs
    - Runtime switching between models via prev_model()/next_model()
    - Automatic observation adaptation when models have different input dimensions
    - Unified predict() interface that handles dimension mismatches
    
    Example:
        runner = MultiModelRunner()
        runner.add_model(model1, "baseline", "logs/baseline", obs_dim=45)
        runner.add_model(model2, "improved", "logs/improved", obs_dim=54)
        
        # Get action from current model
        action, _ = runner.predict(observation)
        
        # Switch to next model
        runner.next_model()
    """
    
    def __init__(self):
        """Initialize the multi-model runner."""
        self.models: List[Any] = []          # List of loaded SB3 models
        self.model_names: List[str] = []     # Display names
        self.log_dirs: List[str] = []        # Source directories
        self.obs_dims: List[int] = []        # Expected observation dimensions
        self.act_dims: List[int] = []        # Expected action dimensions
        self.current_idx: int = 0
        self._warned_mismatch: set = set()   # Track which mismatches we've warned about
    
    def add_model(
        self, 
        model: Any, 
        name: str, 
        log_dir: str, 
        obs_dim: Optional[int] = None, 
        act_dim: Optional[int] = None
    ) -> None:
        """Add a model to the collection.
        
        Args:
            model: Loaded SB3 model instance
            name: Display name for the model
            log_dir: Source directory the model was loaded from
            obs_dim: Expected observation dimension (for adaptation)
            act_dim: Expected action dimension
        """
        self.models.append(model)
        self.model_names.append(name)
        self.log_dirs.append(log_dir)
        self.obs_dims.append(obs_dim)
        self.act_dims.append(act_dim)
    
    @property
    def current_model(self) -> Any:
        """Get the currently selected model."""
        if not self.models:
            return None
        return self.models[self.current_idx]
    
    @property
    def current_name(self) -> str:
        """Get the name of the currently selected model."""
        if not self.model_names:
            return "None"
        return self.model_names[self.current_idx]
    
    @property
    def num_models(self) -> int:
        """Get the number of loaded models."""
        return len(self.models)
    
    @property
    def current_obs_dim(self) -> Optional[int]:
        """Get the expected observation dimension for the current model."""
        if not self.obs_dims or self.current_idx >= len(self.obs_dims):
            return None
        return self.obs_dims[self.current_idx]
    
    @property
    def current_act_dim(self) -> Optional[int]:
        """Get the expected action dimension for the current model."""
        if not self.act_dims or self.current_idx >= len(self.act_dims):
            return None
        return self.act_dims[self.current_idx]
    
    def next_model(self) -> str:
        """Switch to the next model.
        
        Returns:
            Name of the new current model
        """
        if self.num_models <= 1:
            return self.current_name
        self.current_idx = (self.current_idx + 1) % self.num_models
        return self.current_name
    
    def prev_model(self) -> str:
        """Switch to the previous model.
        
        Returns:
            Name of the new current model
        """
        if self.num_models <= 1:
            return self.current_name
        self.current_idx = (self.current_idx - 1) % self.num_models
        return self.current_name
    
    def select_model(self, idx: int) -> str:
        """Select a model by index.
        
        Args:
            idx: Index of the model to select
            
        Returns:
            Name of the new current model
        """
        if 0 <= idx < self.num_models:
            self.current_idx = idx
        return self.current_name
    
    def get_status_string(self) -> str:
        """Get a status string for display.
        
        Returns:
            Formatted string like "[1/3] ModelName"
        """
        if self.num_models == 0:
            return "No models loaded"
        return f"[{self.current_idx + 1}/{self.num_models}] {self.current_name}"
    
    def adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """Adapt observation to match the current model's expected dimension.
        
        If the observation is smaller than expected, pad with zeros.
        If larger, truncate.
        
        Args:
            obs: Observation from the environment
            
        Returns:
            Adapted observation matching the model's expected dimension
        """
        expected_dim = self.current_obs_dim
        if expected_dim is None:
            return obs
        
        actual_dim = obs.shape[-1] if len(obs.shape) > 0 else 0
        
        if actual_dim == expected_dim:
            return obs
        
        # Warn once per model about dimension mismatch
        mismatch_key = (self.current_idx, actual_dim, expected_dim)
        if mismatch_key not in self._warned_mismatch:
            self._warned_mismatch.add(mismatch_key)
            action = "Padding" if actual_dim < expected_dim else "Truncating"
            print(f"\n[Warning] Obs dimension mismatch for '{self.current_name}': "
                  f"env={actual_dim}, model={expected_dim}. {action} observation.")
        
        if actual_dim < expected_dim:
            # Pad with zeros
            if len(obs.shape) == 1:
                return np.pad(obs, (0, expected_dim - actual_dim), mode='constant')
            else:
                # Batched observation
                return np.pad(obs, ((0, 0), (0, expected_dim - actual_dim)), mode='constant')
        else:
            # Truncate
            if len(obs.shape) == 1:
                return obs[:expected_dim]
            else:
                return obs[:, :expected_dim]
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """Get action from current model with automatic observation adaptation.
        
        Args:
            obs: Observation from the environment
            deterministic: Use deterministic action (no exploration noise)
            
        Returns:
            Tuple of (action, state) from the model
        """
        model = self.current_model
        if model is None:
            # Return zeros if no model
            act_dim = self.current_act_dim or 3
            return np.zeros(act_dim), None
        
        # Adapt observation to match model's expected dimension
        adapted_obs = self.adapt_observation(obs)
        
        try:
            return model.predict(adapted_obs, deterministic=deterministic)
        except Exception as e:
            print(f"\n[Error] Prediction failed for '{self.current_name}': {e}")
            act_dim = self.current_act_dim or 3
            return np.zeros(act_dim), None
    
    def get_model_list(self) -> str:
        """Get a formatted list of all models.
        
        Returns:
            Multi-line string with model details
        """
        if self.num_models == 0:
            return "No models loaded"
        
        lines = ["", "=" * 50, "Loaded Models:", "=" * 50]
        for i, (name, log_dir, obs_dim) in enumerate(
            zip(self.model_names, self.log_dirs, self.obs_dims)
        ):
            marker = "►" if i == self.current_idx else " "
            dim_str = f"obs={obs_dim}" if obs_dim else ""
            lines.append(f"  {marker} [{i + 1}] {name} ({dim_str})")
            lines.append(f"       {log_dir}")
        lines.append("=" * 50)
        lines.append("Keys: , = prev, . = next, / = this list")
        lines.append("=" * 50)
        return "\n".join(lines)


def find_checkpoint_path(
    log_path: Path, 
    checkpoint: Optional[str] = "latest"
) -> Optional[str]:
    """Find checkpoint file path in log directory.
    
    Searches both the log directory root and a 'checkpoints/' subdirectory.
    
    Args:
        log_path: Path to log directory
        checkpoint: Checkpoint specifier:
            - "latest" or None: Find the latest checkpoint by step number
            - "best": Find best_model.zip
            - "final": Find final_model.zip
            - A step number string (e.g., "500000"): Find that specific checkpoint
            - A path: Use directly if exists
        
    Returns:
        Path to checkpoint file or None if not found
    """
    # Check both checkpoints subdirectory and root log directory
    search_dirs = [log_path / "checkpoints", log_path]
    
    if checkpoint is None or checkpoint == "latest":
        # Find latest checkpoint by step number
        for search_dir in search_dirs:
            pattern = str(search_dir / "rl_model_*_steps.zip")
            checkpoint_files = glob.glob(pattern)
            if checkpoint_files:
                # Extract step numbers and find max
                def get_steps(f):
                    try:
                        return int(os.path.basename(f).split("_")[2])
                    except (IndexError, ValueError):
                        return 0
                checkpoint_files.sort(key=get_steps, reverse=True)
                return checkpoint_files[0]
        
        # Try final_model.zip in both locations
        for search_dir in search_dirs:
            final_path = search_dir / "final_model.zip"
            if final_path.exists():
                return str(final_path)
            
    elif checkpoint == "best":
        for search_dir in search_dirs:
            best_path = search_dir / "best_model.zip"
            if best_path.exists():
                return str(best_path)
            
    elif checkpoint == "final":
        for search_dir in search_dirs:
            final_path = search_dir / "final_model.zip"
            if final_path.exists():
                return str(final_path)
            
    elif checkpoint.isdigit():
        # Specific step number
        for search_dir in search_dirs:
            checkpoint_path = search_dir / f"rl_model_{checkpoint}_steps.zip"
            if checkpoint_path.exists():
                return str(checkpoint_path)
    else:
        # Assume it's a path
        if os.path.exists(checkpoint):
            return checkpoint
        checkpoint_path = log_path / checkpoint
        if checkpoint_path.exists():
            return str(checkpoint_path)
    
    return None


def _create_dummy_env(
    obs_dim: int, 
    act_dim: int, 
    act_low: float = -1.0, 
    act_high: float = 1.0
):
    """Create a minimal dummy environment for model loading.
    
    Some SB3 algorithms (like CrossQ from sb3-contrib) require an environment
    to be passed during loading. This creates a minimal compatible environment.
    
    Args:
        obs_dim: Observation space dimension
        act_dim: Action space dimension
        act_low: Lower bound for action space
        act_high: Upper bound for action space
        
    Returns:
        A minimal gymnasium environment with the specified spaces
    """
    import gymnasium as gym
    from gymnasium import spaces
    
    class DummyEnv(gym.Env):
        def __init__(self, obs_dim, act_dim, act_low, act_high):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=act_low, high=act_high, shape=(act_dim,), dtype=np.float32
            )
        
        def reset(self, **kwargs):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        
        def step(self, action):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}
    
    return DummyEnv(obs_dim, act_dim, act_low, act_high)


def load_model_standalone(
    checkpoint_path: str, 
    obs_dim: Optional[int] = None, 
    act_dim: Optional[int] = None,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[Any, Optional[int], Optional[int]]:
    """Load an SB3 model without requiring an environment.
    
    This function creates a minimal dummy environment to satisfy SB3's loading
    requirements, inferring the observation/action dimensions from the checkpoint
    metadata when possible.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        obs_dim: Observation dimension (optional, inferred from checkpoint)
        act_dim: Action dimension (optional, inferred from checkpoint)
        device: Device to load model on ("auto", "cuda", "cpu")
        verbose: If True, print loading progress
        
    Returns:
        Tuple of (model, obs_dim, act_dim) or (None, None, None) if loading failed
    """
    if verbose:
        print(f"    Checkpoint: {checkpoint_path}")
    
    # Default action bounds
    act_low = -1.0
    act_high = 1.0
    
    # Try to infer dimensions and bounds from the checkpoint
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            if 'data' in zf.namelist():
                with zf.open('data') as f:
                    data = json.load(f)
                    if 'observation_space' in data:
                        obs_space = data['observation_space']
                        if '_shape' in obs_space:
                            obs_dim = obs_space['_shape'][0]
                    if 'action_space' in data:
                        act_space = data['action_space']
                        if '_shape' in act_space:
                            act_dim = act_space['_shape'][0]
                        # Get action bounds from low_repr/high_repr
                        if 'low_repr' in act_space:
                            try:
                                act_low = float(act_space['low_repr'])
                            except ValueError:
                                pass
                        if 'high_repr' in act_space:
                            try:
                                act_high = float(act_space['high_repr'])
                            except ValueError:
                                pass
    except Exception as e:
        if verbose:
            print(f"    Could not infer dimensions: {e}")
    
    # Default dimensions if still unknown
    if obs_dim is None:
        obs_dim = 100  # Reasonable default
    if act_dim is None:
        act_dim = 3    # Common for tripod
    
    if verbose:
        print(f"    Dims: obs={obs_dim}, act={act_dim}, bounds=[{act_low}, {act_high}]")
    
    # Create dummy env for algorithms that need it
    dummy_env = _create_dummy_env(obs_dim, act_dim, act_low, act_high)
    
    # Mapping of algorithms to try
    algorithms = [
        ("sb3_contrib", "CrossQ"),
        ("sb3_contrib", "TQC"),
        ("stable_baselines3", "SAC"),
        ("stable_baselines3", "PPO"),
        ("stable_baselines3", "TD3"),
    ]
    
    last_error = None
    for module_name, class_name in algorithms:
        try:
            import importlib
            module = importlib.import_module(module_name)
            algo_cls = getattr(module, class_name)
            # Try loading with dummy env first (needed for CrossQ)
            model = algo_cls.load(checkpoint_path, env=dummy_env, device=device)
            if verbose:
                print(f"    Algorithm: {class_name}")
            return model, obs_dim, act_dim
        except Exception as e:
            last_error = e
            # Try without env as fallback
            try:
                model = algo_cls.load(checkpoint_path, env=None, device=device)
                if verbose:
                    print(f"    Algorithm: {class_name} (no env)")
                return model, obs_dim, act_dim
            except Exception:
                pass
            continue
    
    # Print the last error for debugging
    if last_error and verbose:
        print(f"    Last error: {last_error}")
    
    return None, None, None


def load_multiple_models(
    log_dirs: List[str], 
    model_names: Optional[List[str]] = None, 
    checkpoint: str = "latest",
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[MultiModelRunner, Any]:
    """Load multiple SB3 models from log directories.
    
    This function loads multiple trained models into a MultiModelRunner,
    enabling runtime switching between different policies.
    
    Args:
        log_dirs: List of log directory paths containing config.yaml and checkpoints
        model_names: Optional list of display names (default: directory names)
        checkpoint: Checkpoint selection for all models ("latest", "best", "final", or step)
        device: Device to load models on ("auto", "cuda", "cpu")
        verbose: If True, print loading progress
        
    Returns:
        Tuple of (MultiModelRunner with all loaded models, first config object)
        The first config can be used to create the environment.
        
    Example:
        runner, first_cfg = load_multiple_models([
            "logs/baseline_experiment",
            "logs/improved_experiment",
        ])
        
        # Create environment from first config
        first_cfg.environment.mode = "real"
        env = RealMetaMachine(first_cfg)
        
        # Run with model switching
        while running:
            action, _ = runner.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
    """
    # Lazy import to avoid circular dependency
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    runner = MultiModelRunner()
    first_cfg = None
    
    # Generate default names if not provided
    if model_names is None:
        model_names = [os.path.basename(os.path.normpath(d)) for d in log_dirs]
    
    # Ensure we have enough names
    while len(model_names) < len(log_dirs):
        model_names.append(f"model_{len(model_names) + 1}")
    
    if verbose:
        print("=" * 60)
        print(f"Loading {len(log_dirs)} models...")
        print("=" * 60)
    
    for i, (log_dir, name) in enumerate(zip(log_dirs, model_names)):
        if verbose:
            print(f"\n[{i + 1}/{len(log_dirs)}] Loading: {name}")
            print(f"    From: {log_dir}")
        
        try:
            log_path = Path(log_dir)
            
            # Load config (needed for first model to create env)
            config_path = log_path / "config.yaml"
            if not config_path.exists():
                if verbose:
                    print(f"    ✗ Config file not found: {config_path}")
                continue
            
            cfg = ConfigRegistry.create_from_file(str(config_path))
            
            # Keep first config for environment creation
            if first_cfg is None:
                first_cfg = cfg
            
            # Find checkpoint path
            checkpoint_path = find_checkpoint_path(log_path, checkpoint)
            if checkpoint_path is None:
                if verbose:
                    print(f"    ✗ No checkpoint found")
                continue
            
            # Load model without env
            model, obs_dim, act_dim = load_model_standalone(
                checkpoint_path, device=device, verbose=verbose
            )
            
            if model is not None:
                runner.add_model(model, name, log_dir, obs_dim=obs_dim, act_dim=act_dim)
                if verbose:
                    print(f"    ✓ Loaded successfully")
            else:
                if verbose:
                    print(f"    ✗ Failed to load model")
                
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed to load: {e}")
                import traceback
                traceback.print_exc()
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Successfully loaded {runner.num_models}/{len(log_dirs)} models")
        
        # Show observation dimensions if they differ
        if runner.num_models > 1:
            unique_dims = set(runner.obs_dims)
            if len(unique_dims) > 1:
                print("\n⚠️  Note: Models have different observation dimensions!")
                print("   Observations will be automatically padded/truncated when switching.")
                for name, obs_dim in zip(runner.model_names, runner.obs_dims):
                    print(f"     - {name}: obs_dim={obs_dim}")
        
        print("=" * 60)
    
    return runner, first_cfg

