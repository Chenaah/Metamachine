"""
MetaMachine Environments Module

This module provides simulation and real robot environments for the MetaMachine framework.

Core Classes:
    - MetaMachine: Main simulation environment (env_sim.py)
    - RealMetaMachine: Real robot environment using capybarish (env_real.py)
    - RayVecMetaMachine: Vectorized environment using Ray for parallel execution (vec_env.py)
    - VecEnv: Abstract base class for vectorized environments (vec_env.py)

Factory Function:
    - make_env(cfg): Creates the appropriate environment based on config mode

Example:
    >>> from metamachine.environments import make_env
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> # Load config with mode: "sim" or "real"
    >>> cfg = ConfigRegistry.create_from_file("my_config.yaml")
    >>> 
    >>> # Factory automatically creates correct environment type
    >>> env = make_env(cfg)
    >>> 
    >>> # Or use specific classes directly
    >>> from metamachine.environments import MetaMachine, RealMetaMachine
    >>> sim_env = MetaMachine(cfg)
    >>> real_env = RealMetaMachine(cfg)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from .env_sim import MetaMachine
from .base import Base

# Optional import for real robot environment (requires capybarish)
try:
    from .env_real import RealMetaMachine
except ImportError:
    RealMetaMachine = None

# Optional imports for vectorized environments (require Ray)
try:
    from .vec_env import RayVecMetaMachine, VecEnv, StateSnapshot
except ImportError:
    # Ray not available
    RayVecMetaMachine = None
    VecEnv = None
    StateSnapshot = None


def make_env(cfg, **kwargs):
    """Factory function to create the appropriate environment based on config.
    
    This function checks `cfg.environment.mode` and creates either:
    - MetaMachine (simulation) if mode == "sim" or not specified
    - RealMetaMachine (real robot) if mode == "real"
    
    Args:
        cfg: Configuration object (OmegaConf) with environment settings
        **kwargs: Additional arguments passed to the environment constructor
    
    Returns:
        Environment instance (MetaMachine or RealMetaMachine)
    
    Raises:
        ValueError: If mode is "real" but capybarish is not installed
        ValueError: If mode is unknown
    
    Example:
        >>> from metamachine.environments import make_env
        >>> from metamachine.environments.configs.config_registry import ConfigRegistry
        >>> 
        >>> # For simulation
        >>> cfg = ConfigRegistry.create_from_file("config.yaml")
        >>> cfg.environment.mode = "sim"
        >>> sim_env = make_env(cfg)
        >>> 
        >>> # For real robot
        >>> cfg.environment.mode = "real"
        >>> real_env = make_env(cfg)
    """
    # Get mode from config
    mode = cfg.environment.get("mode", "sim").lower()
    
    if mode == "sim" or mode == "simulation":
        return MetaMachine(cfg, **kwargs)
    
    elif mode == "real":
        if RealMetaMachine is None:
            raise ValueError(
                "Real robot mode requires capybarish. "
                "Install it with: pip install capybarish"
            )
        return RealMetaMachine(cfg, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown environment mode: '{mode}'. "
            f"Use 'sim' for simulation or 'real' for real robot."
        )


__all__ = [
    # Core environments
    "MetaMachine",
    "RealMetaMachine",
    "Base",
    
    # Factory function
    "make_env",
    
    # Vectorized environments
    "RayVecMetaMachine",
    "VecEnv",
    "StateSnapshot",
]
