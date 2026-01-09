"""
Policy Runner Utilities for MetaMachine

This module provides utilities for loading and managing multiple trained policies
for comparison, A/B testing, or runtime policy switching.

Features:
- PolicyRunner: Manages multiple loaded SB3 policies with easy switching
- load_policies: Load multiple policies from different log directories
- Automatic observation adaptation for policies with different observation dimensions
- Support for all SB3-compatible algorithms (CrossQ, SAC, PPO, etc.)

Example usage:

    from metamachine.utils.policy_runner import load_policies

    # Load multiple trained policies
    runner, first_cfg = load_policies([
        "logs/experiment1",
        "logs/experiment2",
        "logs/experiment3",
    ])

    # Use in real robot control
    env = RealMetaMachine(first_cfg)
    env.policy_runner = runner

    # Get predictions with automatic policy switching
    action, _ = runner.predict(observation)

    # Switch policies
    runner.next_policy()  # or runner.prev_policy()

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
    "PolicyRunner",
    "load_policies",
    "load_policy_standalone",
    "find_checkpoint_path",
    # Legacy aliases
    "MultiModelRunner",
    "load_multiple_models",
    "load_model_standalone",
]


class PolicyRunner:
    """Manages multiple loaded policies for easy switching during runtime.
    
    This class enables:
    - Loading multiple SB3 policies from different training runs
    - Runtime switching between policies via prev_policy()/next_policy()
    - Automatic observation adaptation when policies have different input dimensions
    - Unified predict() interface that handles dimension mismatches
    
    Example:
        runner = PolicyRunner()
        runner.add_policy(policy1, "baseline", "logs/baseline", obs_dim=45)
        runner.add_policy(policy2, "improved", "logs/improved", obs_dim=54)
        
        # Get action from current policy
        action, _ = runner.predict(observation)
        
        # Switch to next policy
        runner.next_policy()
    """
    
    def __init__(self):
        """Initialize the policy runner."""
        self.policies: List[Any] = []        # List of loaded SB3 policies
        self.policy_names: List[str] = []    # Display names
        self.log_dirs: List[str] = []        # Source directories
        self.obs_dims: List[int] = []        # Expected observation dimensions
        self.act_dims: List[int] = []        # Expected action dimensions
        self.current_idx: int = 0
        self._warned_mismatch: set = set()   # Track which mismatches we've warned about
    
    def add_policy(
        self, 
        policy: Any, 
        name: str, 
        log_dir: str, 
        obs_dim: Optional[int] = None, 
        act_dim: Optional[int] = None
    ) -> None:
        """Add a policy to the collection.
        
        Args:
            policy: Loaded SB3 model/policy instance
            name: Display name for the policy
            log_dir: Source directory the policy was loaded from
            obs_dim: Expected observation dimension (for adaptation)
            act_dim: Expected action dimension
        """
        self.policies.append(policy)
        self.policy_names.append(name)
        self.log_dirs.append(log_dir)
        self.obs_dims.append(obs_dim)
        self.act_dims.append(act_dim)
    
    # Alias for backward compatibility
    def add_model(self, model, name, log_dir, obs_dim=None, act_dim=None):
        """Alias for add_policy() for backward compatibility."""
        return self.add_policy(model, name, log_dir, obs_dim, act_dim)
    
    @property
    def current_policy(self) -> Any:
        """Get the currently selected policy."""
        if not self.policies:
            return None
        return self.policies[self.current_idx]
    
    @property
    def current_model(self) -> Any:
        """Alias for current_policy for backward compatibility."""
        return self.current_policy
    
    @property
    def current_name(self) -> str:
        """Get the name of the currently selected policy."""
        if not self.policy_names:
            return "None"
        return self.policy_names[self.current_idx]
    
    @property
    def num_policies(self) -> int:
        """Get the number of loaded policies."""
        return len(self.policies)
    
    @property
    def num_models(self) -> int:
        """Alias for num_policies for backward compatibility."""
        return self.num_policies
    
    @property
    def current_obs_dim(self) -> Optional[int]:
        """Get the expected observation dimension for the current policy."""
        if not self.obs_dims or self.current_idx >= len(self.obs_dims):
            return None
        return self.obs_dims[self.current_idx]
    
    @property
    def current_act_dim(self) -> Optional[int]:
        """Get the expected action dimension for the current policy."""
        if not self.act_dims or self.current_idx >= len(self.act_dims):
            return None
        return self.act_dims[self.current_idx]
    
    def next_policy(self) -> str:
        """Switch to the next policy.
        
        Returns:
            Name of the new current policy
        """
        if self.num_policies <= 1:
            return self.current_name
        self.current_idx = (self.current_idx + 1) % self.num_policies
        return self.current_name
    
    def next_model(self) -> str:
        """Alias for next_policy() for backward compatibility."""
        return self.next_policy()
    
    def prev_policy(self) -> str:
        """Switch to the previous policy.
        
        Returns:
            Name of the new current policy
        """
        if self.num_policies <= 1:
            return self.current_name
        self.current_idx = (self.current_idx - 1) % self.num_policies
        return self.current_name
    
    def prev_model(self) -> str:
        """Alias for prev_policy() for backward compatibility."""
        return self.prev_policy()
    
    def select_policy(self, idx: int) -> str:
        """Select a policy by index.
        
        Args:
            idx: Index of the policy to select
            
        Returns:
            Name of the new current policy
        """
        if 0 <= idx < self.num_policies:
            self.current_idx = idx
        return self.current_name
    
    def select_model(self, idx: int) -> str:
        """Alias for select_policy() for backward compatibility."""
        return self.select_policy(idx)
    
    def get_status_string(self) -> str:
        """Get a status string for display.
        
        Returns:
            Formatted string like "[1/3] PolicyName"
        """
        if self.num_policies == 0:
            return "No policies loaded"
        return f"[{self.current_idx + 1}/{self.num_policies}] {self.current_name}"
    
    # Backward compatibility aliases for internal attributes
    @property
    def models(self) -> List[Any]:
        """Alias for policies for backward compatibility."""
        return self.policies
    
    @property
    def model_names(self) -> List[str]:
        """Alias for policy_names for backward compatibility."""
        return self.policy_names
    
    def adapt_observation(self, obs: np.ndarray) -> np.ndarray:
        """Adapt observation to match the current policy's expected dimension.
        
        If the observation is smaller than expected, pad with zeros.
        If larger, truncate.
        
        Args:
            obs: Observation from the environment
            
        Returns:
            Adapted observation matching the policy's expected dimension
        """
        expected_dim = self.current_obs_dim
        if expected_dim is None:
            return obs
        
        actual_dim = obs.shape[-1] if len(obs.shape) > 0 else 0
        
        if actual_dim == expected_dim:
            return obs
        
        # Warn once per policy about dimension mismatch
        mismatch_key = (self.current_idx, actual_dim, expected_dim)
        if mismatch_key not in self._warned_mismatch:
            self._warned_mismatch.add(mismatch_key)
            action = "Padding" if actual_dim < expected_dim else "Truncating"
            print(f"\n[Warning] Obs dimension mismatch for '{self.current_name}': "
                  f"env={actual_dim}, policy={expected_dim}. {action} observation.")
        
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
        """Get action from current policy with automatic observation adaptation.
        
        Args:
            obs: Observation from the environment
            deterministic: Use deterministic action (no exploration noise)
            
        Returns:
            Tuple of (action, state) from the policy
        """
        policy = self.current_policy
        if policy is None:
            # Return zeros if no policy
            act_dim = self.current_act_dim or 3
            return np.zeros(act_dim), None
        
        # Adapt observation to match policy's expected dimension
        adapted_obs = self.adapt_observation(obs)
        
        try:
            return policy.predict(adapted_obs, deterministic=deterministic)
        except Exception as e:
            print(f"\n[Error] Prediction failed for '{self.current_name}': {e}")
            act_dim = self.current_act_dim or 3
            return np.zeros(act_dim), None
    
    def get_policy_list(self) -> str:
        """Get a formatted list of all policies.
        
        Returns:
            Multi-line string with policy details
        """
        if self.num_policies == 0:
            return "No policies loaded"
        
        lines = ["", "=" * 50, "Loaded Policies:", "=" * 50]
        for i, (name, log_dir, obs_dim) in enumerate(
            zip(self.policy_names, self.log_dirs, self.obs_dims)
        ):
            marker = "►" if i == self.current_idx else " "
            dim_str = f"obs={obs_dim}" if obs_dim else ""
            lines.append(f"  {marker} [{i + 1}] {name} ({dim_str})")
            lines.append(f"       {log_dir}")
        lines.append("=" * 50)
        lines.append("Keys: , = prev, . = next, / = this list")
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def get_model_list(self) -> str:
        """Alias for get_policy_list() for backward compatibility."""
        return self.get_policy_list()


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


def load_policy_standalone(
    checkpoint_path: str, 
    obs_dim: Optional[int] = None, 
    act_dim: Optional[int] = None,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[Any, Optional[int], Optional[int]]:
    """Load an SB3 policy without requiring an environment.
    
    This function creates a minimal dummy environment to satisfy SB3's loading
    requirements, inferring the observation/action dimensions from the checkpoint
    metadata when possible.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        obs_dim: Observation dimension (optional, inferred from checkpoint)
        act_dim: Action dimension (optional, inferred from checkpoint)
        device: Device to load policy on ("auto", "cuda", "cpu")
        verbose: If True, print loading progress
        
    Returns:
        Tuple of (policy, obs_dim, act_dim) or (None, None, None) if loading failed
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


def load_policies(
    log_dirs: List[str], 
    policy_names: Optional[List[str]] = None, 
    checkpoint: str = "latest",
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[PolicyRunner, Any]:
    """Load multiple SB3 policies from log directories.
    
    This function loads multiple trained policies into a PolicyRunner,
    enabling runtime switching between different policies.
    
    Args:
        log_dirs: List of log directory paths containing config.yaml and checkpoints
        policy_names: Optional list of display names (default: directory names)
        checkpoint: Checkpoint selection for all policies ("latest", "best", "final", or step)
        device: Device to load policies on ("auto", "cuda", "cpu")
        verbose: If True, print loading progress
        
    Returns:
        Tuple of (PolicyRunner with all loaded policies, first config object)
        The first config can be used to create the environment.
        
    Example:
        runner, first_cfg = load_policies([
            "logs/baseline_experiment",
            "logs/improved_experiment",
        ])
        
        # Create environment from first config
        first_cfg.environment.mode = "real"
        env = RealMetaMachine(first_cfg)
        
        # Run with policy switching
        while running:
            action, _ = runner.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
    """
    # Lazy import to avoid circular dependency
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    runner = PolicyRunner()
    first_cfg = None
    
    # Generate default names if not provided
    if policy_names is None:
        policy_names = [os.path.basename(os.path.normpath(d)) for d in log_dirs]
    
    # Ensure we have enough names
    while len(policy_names) < len(log_dirs):
        policy_names.append(f"policy_{len(policy_names) + 1}")
    
    if verbose:
        print("=" * 60)
        print(f"Loading {len(log_dirs)} policies...")
        print("=" * 60)
    
    for i, (log_dir, name) in enumerate(zip(log_dirs, policy_names)):
        if verbose:
            print(f"\n[{i + 1}/{len(log_dirs)}] Loading: {name}")
            print(f"    From: {log_dir}")
        
        try:
            log_path = Path(log_dir)
            
            # Load config (needed for first policy to create env)
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
            
            # Load policy without env
            policy, obs_dim, act_dim = load_policy_standalone(
                checkpoint_path, device=device, verbose=verbose
            )
            
            if policy is not None:
                runner.add_policy(policy, name, log_dir, obs_dim=obs_dim, act_dim=act_dim)
                if verbose:
                    print(f"    ✓ Loaded successfully")
            else:
                if verbose:
                    print(f"    ✗ Failed to load policy")
                
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed to load: {e}")
                import traceback
                traceback.print_exc()
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"Successfully loaded {runner.num_policies}/{len(log_dirs)} policies")
        
        # Show observation dimensions if they differ
        if runner.num_policies > 1:
            unique_dims = set(runner.obs_dims)
            if len(unique_dims) > 1:
                print("\n⚠️  Note: Policies have different observation dimensions!")
                print("   Observations will be automatically padded/truncated when switching.")
                for name, obs_dim in zip(runner.policy_names, runner.obs_dims):
                    print(f"     - {name}: obs_dim={obs_dim}")
        
        print("=" * 60)
    
    return runner, first_cfg


# Backward compatibility aliases
def load_multiple_models(log_dirs, model_names=None, checkpoint="latest", device="auto", verbose=True):
    """Alias for load_policies() for backward compatibility."""
    return load_policies(log_dirs, policy_names=model_names, checkpoint=checkpoint, device=device, verbose=verbose)


def load_model_standalone(checkpoint_path, obs_dim=None, act_dim=None, device="auto", verbose=True):
    """Alias for load_policy_standalone() for backward compatibility."""
    return load_policy_standalone(checkpoint_path, obs_dim=obs_dim, act_dim=act_dim, device=device, verbose=verbose)


# Backward compatibility class alias
MultiModelRunner = PolicyRunner

