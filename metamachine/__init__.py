"""
MetaMachine - A Modular Robotic Simulation Framework
MetaMachine is a simulation framework for modular robots. It provides flexible
environments and tools for reinforcement learning research, evolutionary robotics
research, and development of adaptive robotic systems with modular morphologies.
Key modules:
- environments: Core simulation environments and components
- robot_factory: Robot design and morphology generation tools
- utils: Utility functions and helper classes
Example usage:
from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine
cfg = ConfigRegistry.create_from_name("basic_quadruped")
env = MetaMachine(cfg)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

# Package version
__version__ = "0.1.1"

# Root directory for asset and configuration file paths
METAMACHINE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_asset_path(*path_components: str) -> str:
    """
    Get the path to an asset file with robust path resolution.

    This function tries multiple locations to find assets:
    1. In the installed package location
    2. In the development source tree
    3. Using importlib.resources (for proper package installations)

    Args:
        *path_components: Path components relative to the assets directory

    Returns:
        str: Full path to the asset file

    Raises:
        FileNotFoundError: If the asset cannot be found in any location
    """
    # Try the direct path first (development mode)
    direct_path = os.path.join(METAMACHINE_ROOT_DIR, "assets", *path_components)
    if os.path.exists(direct_path):
        return direct_path

    # Try using importlib.resources for proper package installations
    try:
        import importlib.resources as resources

        # For Python 3.9+, use files() API
        if hasattr(resources, "files"):
            asset_path = resources.files("metamachine.assets").joinpath(
                *path_components
            )
            if hasattr(asset_path, "exists") and asset_path.exists():
                return str(asset_path)
        # Fallback for older Python versions
        else:
            with resources.path("metamachine.assets", path_components[-1]) as path:
                return str(path)
    except (ImportError, AttributeError, FileNotFoundError):
        pass

    # Last resort: check if we're in a source tree
    source_path = os.path.join(
        os.path.dirname(METAMACHINE_ROOT_DIR), "metamachine", "assets", *path_components
    )
    if os.path.exists(source_path):
        return source_path

    # If nothing works, return the direct path and let the caller handle the error
    return direct_path


# Package metadata
__author__ = "Chen Yu"
__email__ = "chenyu@u.northwestern.edu"
__description__ = "A simulation framework for modular robots"
