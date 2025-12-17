"""
MetaMachine Environments Module

This module provides simulation and real robot environments for the MetaMachine framework.

Core Classes:
    - MetaMachine: Main simulation environment (env_sim.py)
    - RayVecMetaMachine: Vectorized environment using Ray for parallel execution (vec_env.py)
    - VecEnv: Abstract base class for vectorized environments (vec_env.py)

Example:
    >>> from metamachine.environments import MetaMachine, RayVecMetaMachine
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> # Single environment
    >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
    >>> env = MetaMachine(cfg)
    >>> 
    >>> # Vectorized environment (requires Ray)
    >>> vec_env = RayVecMetaMachine(cfg, num_envs=8)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from .env_sim import MetaMachine
from .base import Base

# Optional imports for vectorized environments (require Ray)
try:
    from .vec_env import RayVecMetaMachine, VecEnv, StateSnapshot
except ImportError:
    # Ray not available
    RayVecMetaMachine = None
    VecEnv = None
    StateSnapshot = None

__all__ = [
    "MetaMachine",
    "Base",
    "RayVecMetaMachine",
    "VecEnv",
    "StateSnapshot",
]

