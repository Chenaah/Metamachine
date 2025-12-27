"""
Metamachine Robot Factory Module

This module provides the core functionality for generating MuJoCo XML files
from modular robot configurations using a modern factory pattern.

Built-in factory:
- **modular_legs**: Tree-based morphology using sequential module connections

Plugin factories (load separately):
- **lego_legs**: Graph-based morphology using weld constraints between components

All factories share a unified API through the BaseRobotFactory interface.

Quick Start:
    # Using the registry (recommended)
    >>> from metamachine.robot_factory import get_robot_factory
    >>> factory = get_robot_factory("modular_legs")
    >>> robot = factory.create_robot(...)
    
    # Using graph-based morphology
    >>> from metamachine.robot_factory.morphology import create_tripod_graph
    >>> morphology = create_tripod_graph()

Plugin System:
    Factories can be loaded as plugins from external directories:
    >>> from metamachine.robot_factory import load_plugins_from
    >>> load_plugins_from("/path/to/private_plugins")
    >>> factory = get_robot_factory("lego_legs")  # Now available!
    
Random Generation:
    The random generation API provides abstract classes for generating random
    robot morphologies. Factory-specific implementations are in plugins:
    
    # Abstract classes (core)
    >>> from metamachine.robot_factory import (
    ...     ComponentBudget,
    ...     ComponentDockRegistry,
    ...     RandomGenerationCapability,
    ... )
    
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

# Import base classes for custom factory development
from .base_factory import BaseRobot, BaseRobotFactory, RobotSpec, RobotType

# Import the factory registry system
from .factory_registry import (
    get_factories_by_type,
    get_registry,
    get_robot_factory,
    list_factories,
    register_factory,
    search_factories,
    load_plugins_from,
    list_plugins,
)

# Import morphology classes (unified graph-based representation)
from .morphology import (
    ComponentSpec,
    ComponentType,
    Connection,
    RobotGraph,
    create_tripod_graph,
    create_quadruped_graph,
)

# Import random generation API (abstract layer only)
from .random_generation import (
    # Dock system (generic)
    ComponentDockRegistry,
    DockGender,
    DockSpec,
    # Budget (generic base)
    ComponentBudget,
    # Generation (generic)
    GenerationConfig,
    GenerationMode,
    BaseRandomGraphGenerator,
    # Factory integration
    RandomGenerationCapability,
    # Utilities
    graph_to_yaml_dict,
    visualize_graph,
)

# Legacy compatibility imports
from .modular_legs.meta_designer import ModularLegs

# Import specific factories
from .modular_legs.modular_legs_factory import ModularLegsFactory, ModularLegsRobot

# Export public API
__all__ = [
    # Factory registry functions
    "get_robot_factory",
    "register_factory",
    "list_factories",
    "get_factories_by_type",
    "search_factories",
    "get_registry",
    # Plugin system
    "load_plugins_from",
    "list_plugins",
    # Base classes
    "BaseRobotFactory",
    "BaseRobot",
    "RobotType",
    "RobotSpec",
    # Morphology classes
    "ComponentSpec",
    "ComponentType",
    "Connection",
    "RobotGraph",
    "create_tripod_graph",
    "create_quadruped_graph",
    # Random generation (abstract layer)
    "ComponentBudget",
    "ComponentDockRegistry",
    "DockGender",
    "DockSpec",
    "GenerationConfig",
    "GenerationMode",
    "BaseRandomGraphGenerator",
    "RandomGenerationCapability",
    "graph_to_yaml_dict",
    "visualize_graph",
    # ModularLegs factory
    "ModularLegsFactory",
    "ModularLegsRobot",
    # Legacy compatibility
    "ModularLegs",
]


# Legacy compatibility function
def get_robot_factory_legacy(factory_name: str = "modular_legs"):
    """
    Legacy compatibility function for getting robot factories.

    This function maintains backward compatibility with the old factory system.
    New code should use the registry-based get_robot_factory function.
    """
    import warnings

    warnings.warn(
        "get_robot_factory_legacy is deprecated. Use get_robot_factory instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Map old factory names to new ones
    legacy_mapping = {
        "modular_legs": "modular_legs",
        "mini_modular_legs": "mini_modular_legs",
        "lego_legs": "lego_legs",
    }

    new_name = legacy_mapping.get(factory_name)
    if new_name is None:
        raise ValueError(f"Unknown factory name: {factory_name}")

    return get_robot_factory(new_name)
