"""
Robot factory registry and management system.
This module provides a centralized system for registering, discovering, and
instantiating robot factories. It supports both built-in and custom factories
with validation and capability discovery.

The registry also supports a plugin system for dynamic factory loading,
allowing factories to be:
- Loaded from separate directories/packages
- Enabled/disabled at runtime
- Kept private/separate from the main codebase

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

import functools
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from metamachine.robot_factory.modular_legs.constants import (
    MESH_DICT_DRAFT,
    MESH_DICT_FINE,
    ROBOT_CFG_AIR1S,
)

from .base_factory import BaseRobotFactory, RobotType

if TYPE_CHECKING:
    from .plugin_loader import PluginLoader

logger = logging.getLogger(__name__)


class FactoryPriority(Enum):
    """Priority levels for factory registration."""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class FactoryRegistration:
    """Registration information for a robot factory."""

    factory_class: type[BaseRobotFactory]
    name: str
    robot_type: RobotType
    priority: FactoryPriority = FactoryPriority.MEDIUM
    description: str = ""
    tags: list[str] = field(default_factory=list)
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True


class RobotFactoryRegistry:
    """
    Registry for managing robot factories.

    This class provides a centralized system for registering, discovering, and
    instantiating robot factories. It supports validation, capability discovery,
    and flexible factory selection.
    """

    def __init__(self) -> None:
        """Initialize the factory registry."""
        self._factories: dict[str, FactoryRegistration] = {}
        self._type_mapping: dict[RobotType, list[str]] = {}
        self._tag_mapping: dict[str, list[str]] = {}

        # Register built-in factories
        self._register_builtin_factories()

    def register_factory(
        self,
        factory_class: type[BaseRobotFactory],
        name: str,
        robot_type: RobotType,
        priority: FactoryPriority = FactoryPriority.MEDIUM,
        description: str = "",
        tags: Optional[list[str]] = None,
        default_kwargs: Optional[dict[str, Any]] = None,
        is_enabled: bool = True,
        replace_existing: bool = False,
    ) -> bool:
        """
        Register a robot factory.

        Args:
            factory_class: Factory class to register
            name: Unique name for the factory
            robot_type: Type of robot this factory creates
            priority: Priority level for factory selection
            description: Description of the factory
            tags: Tags for categorization and search
            default_kwargs: Default keyword arguments for factory instantiation
            is_enabled: Whether the factory is enabled
            replace_existing: Whether to replace existing registration

        Returns:
            bool: True if registration was successful, False otherwise
        """
        if name in self._factories and not replace_existing:
            logger.warning(
                f"Factory '{name}' already registered. Use replace_existing=True to override."
            )
            return False

        if not self._validate_factory_class(factory_class):
            logger.error(f"Invalid factory class: {factory_class}")
            return False

        registration = FactoryRegistration(
            factory_class=factory_class,
            name=name,
            robot_type=robot_type,
            priority=priority,
            description=description,
            tags=tags or [],
            default_kwargs=default_kwargs or {},
            is_enabled=is_enabled,
        )

        self._factories[name] = registration
        self._update_mappings(name, robot_type, tags or [])

        logger.info(f"Registered factory '{name}' for robot type '{robot_type.value}'")
        return True

    def unregister_factory(self, name: str) -> bool:
        """
        Unregister a factory.

        Args:
            name: Name of the factory to unregister

        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        if name not in self._factories:
            logger.warning(f"Factory '{name}' not found for unregistration")
            return False

        registration = self._factories.pop(name)
        self._remove_from_mappings(name, registration.robot_type, registration.tags)

        logger.info(f"Unregistered factory '{name}'")
        return True

    def get_factory(self, name: str, **kwargs) -> Optional[BaseRobotFactory]:
        """
        Get a factory instance by name.

        Args:
            name: Name of the factory
            **kwargs: Additional arguments to pass to factory constructor

        Returns:
            BaseRobotFactory instance or None if not found
        """
        if name not in self._factories:
            logger.error(f"Factory '{name}' not found")
            return None

        registration = self._factories[name]
        if not registration.is_enabled:
            logger.warning(f"Factory '{name}' is disabled")
            return None

        try:
            # Merge default kwargs with provided kwargs
            merged_kwargs = registration.default_kwargs.copy()
            merged_kwargs.update(kwargs)

            factory = registration.factory_class(**merged_kwargs)
            return factory
        except Exception as e:
            logger.error(f"Failed to instantiate factory '{name}': {e}")
            return None

    def get_factories_by_type(self, robot_type: RobotType) -> list[BaseRobotFactory]:
        """
        Get all factories for a specific robot type.

        Args:
            robot_type: Robot type to filter by

        Returns:
            List of factory instances
        """
        if robot_type not in self._type_mapping:
            return []

        factories = []
        for name in self._type_mapping[robot_type]:
            factory = self.get_factory(name)
            if factory:
                factories.append(factory)

        # Sort by priority
        factories.sort(key=lambda f: self._factories[f.name].priority.value)
        return factories

    def get_factories_by_tag(self, tag: str) -> list[BaseRobotFactory]:
        """
        Get all factories with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of factory instances
        """
        if tag not in self._tag_mapping:
            return []

        factories = []
        for name in self._tag_mapping[tag]:
            factory = self.get_factory(name)
            if factory:
                factories.append(factory)

        return factories

    def list_factories(self, enabled_only: bool = True) -> list[str]:
        """
        List all registered factory names.

        Args:
            enabled_only: Whether to include only enabled factories

        Returns:
            List of factory names
        """
        if enabled_only:
            return [name for name, reg in self._factories.items() if reg.is_enabled]
        return list(self._factories.keys())

    def get_factory_info(self, name: str) -> Optional[dict[str, Any]]:
        """
        Get information about a specific factory.

        Args:
            name: Name of the factory

        Returns:
            Dictionary with factory information or None if not found
        """
        if name not in self._factories:
            return None

        registration = self._factories[name]
        return {
            "name": name,
            "robot_type": registration.robot_type.value,
            "priority": registration.priority.value,
            "description": registration.description,
            "tags": registration.tags,
            "is_enabled": registration.is_enabled,
            "class": registration.factory_class.__name__,
            "module": registration.factory_class.__module__,
        }

    def get_all_factory_info(self) -> list[dict[str, Any]]:
        """
        Get information about all registered factories.

        Returns:
            List of factory information dictionaries
        """
        return [self.get_factory_info(name) for name in self._factories.keys()]

    def search_factories(
        self,
        query: str,
        robot_type: Optional[RobotType] = None,
        tags: Optional[list[str]] = None,
        enabled_only: bool = True,
    ) -> list[str]:
        """
        Search for factories based on various criteria.

        Args:
            query: Search query (matches name or description)
            robot_type: Filter by robot type
            tags: Filter by tags (must have all specified tags)
            enabled_only: Whether to include only enabled factories

        Returns:
            List of matching factory names
        """
        matches = []

        for name, registration in self._factories.items():
            if enabled_only and not registration.is_enabled:
                continue

            # Check robot type filter
            if robot_type and registration.robot_type != robot_type:
                continue

            # Check tags filter
            if tags and not all(tag in registration.tags for tag in tags):
                continue

            # Check query match
            if (
                query.lower() in name.lower()
                or query.lower() in registration.description.lower()
            ):
                matches.append(name)

        return matches

    def enable_factory(self, name: str) -> bool:
        """
        Enable a factory.

        Args:
            name: Name of the factory to enable

        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self._factories:
            return False

        self._factories[name].is_enabled = True
        logger.info(f"Enabled factory '{name}'")
        return True

    def disable_factory(self, name: str) -> bool:
        """
        Disable a factory.

        Args:
            name: Name of the factory to disable

        Returns:
            bool: True if successful, False otherwise
        """
        if name not in self._factories:
            return False

        self._factories[name].is_enabled = False
        logger.info(f"Disabled factory '{name}'")
        return True

    def _validate_factory_class(self, factory_class: type[BaseRobotFactory]) -> bool:
        """
        Validate that a class is a proper factory class.

        Args:
            factory_class: Class to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if isinstance(factory_class, functools.partial):
                factory_class_ = factory_class.func
            else:
                factory_class_ = factory_class

            # Check if it's a subclass of BaseRobotFactory
            if not issubclass(factory_class_, BaseRobotFactory):
                return False
            # Check if it's not abstract

            if inspect.isabstract(factory_class_):
                return False

            return True
        except TypeError:
            return False

    def _update_mappings(
        self, name: str, robot_type: RobotType, tags: list[str]
    ) -> None:
        """Update internal mappings for efficient lookup."""
        # Update type mapping
        if robot_type not in self._type_mapping:
            self._type_mapping[robot_type] = []
        self._type_mapping[robot_type].append(name)

        # Update tag mapping
        for tag in tags:
            if tag not in self._tag_mapping:
                self._tag_mapping[tag] = []
            self._tag_mapping[tag].append(name)

    def _remove_from_mappings(self, name: str, robot_type: RobotType, tags: list[str]):
        """Remove from internal mappings."""
        # Remove from type mapping
        if robot_type in self._type_mapping:
            self._type_mapping[robot_type].remove(name)
            if not self._type_mapping[robot_type]:
                del self._type_mapping[robot_type]

        # Remove from tag mapping
        for tag in tags:
            if tag in self._tag_mapping:
                self._tag_mapping[tag].remove(name)
                if not self._tag_mapping[tag]:
                    del self._tag_mapping[tag]

    def _register_builtin_factories(self):
        """Register built-in factories."""
        # Register ModularLegs factory
        try:
            from .modular_legs.modular_legs_factory import ModularLegsFactory

            # Register standard modular legs
            self.register_factory(
                factory_class=ModularLegsFactory,
                name="modular_legs",
                robot_type=RobotType.MODULAR_LEGS,
                priority=FactoryPriority.HIGH,
                description="Standard modular legs robot factory",
                tags=["modular", "legs", "standard", "tree"],
                default_kwargs={},
            )

            self.register_factory(
                factory_class=functools.partial(
                    ModularLegsFactory,
                    robot_cfg=ROBOT_CFG_AIR1S,
                    mesh_dict=MESH_DICT_DRAFT,
                ),
                name="modular_legs_draft",
                robot_type=RobotType.MODULAR_LEGS,
                priority=FactoryPriority.HIGH,
                description="Draft modular legs robot factory",
                tags=["modular", "legs", "draft", "tree"],
                default_kwargs={},
            )

        except ImportError as e:
            logger.warning(f"Could not register built-in factories: {e}")
        except Exception as e:
            logger.error(f"Error registering built-in factories: {e}")


# Global registry instance
_registry = RobotFactoryRegistry()

# Plugin loader instance (lazy initialized)
_plugin_loader: Optional["PluginLoader"] = None


def get_plugin_loader() -> "PluginLoader":
    """
    Get the global plugin loader instance.
    
    Returns:
        The global PluginLoader instance
    """
    global _plugin_loader
    
    if _plugin_loader is None:
        from .plugin_loader import PluginLoader
        _plugin_loader = PluginLoader(_registry, auto_load_builtin=True)
    
    return _plugin_loader


def load_plugins_from(path: str) -> dict[str, bool]:
    """
    Load factory plugins from a directory.
    
    This allows you to keep some factories in separate directories
    that can be easily included or excluded from releases.
    
    Args:
        path: Directory containing plugin packages
        
    Returns:
        Dict mapping plugin names to success status
        
    Example:
        >>> # Load private factories from a separate directory
        >>> load_plugins_from("/path/to/private/factories")
        {'my_private_factory': True}
    """
    loader = get_plugin_loader()
    loader.add_plugin_directory(path)
    return loader.load_all_plugins()


def list_plugins() -> list[dict]:
    """
    List all known plugins and their status.
    
    Returns:
        List of plugin information dictionaries
    """
    return get_plugin_loader().list_plugins()


def get_robot_factory(name: str, **kwargs) -> Optional[BaseRobotFactory]:
    """
    Get a robot factory instance by name.

    Args:
        name: Name of the factory
        **kwargs: Additional arguments to pass to factory constructor

    Returns:
        BaseRobotFactory instance or None if not found
    """
    return _registry.get_factory(name, **kwargs)


def register_factory(
    factory_class: type[BaseRobotFactory], name: str, robot_type: RobotType, **kwargs
) -> bool:
    """
    Register a robot factory in the global registry.

    Args:
        factory_class: Factory class to register
        name: Unique name for the factory
        robot_type: Type of robot this factory creates
        **kwargs: Additional registration parameters

    Returns:
        bool: True if registration was successful, False otherwise
    """
    return _registry.register_factory(factory_class, name, robot_type, **kwargs)


def list_factories(enabled_only: bool = True) -> list[str]:
    """
    List all registered factory names.

    Args:
        enabled_only: Whether to include only enabled factories

    Returns:
        List of factory names
    """
    return _registry.list_factories(enabled_only)


def get_factories_by_type(robot_type: RobotType) -> list[BaseRobotFactory]:
    """
    Get all factories for a specific robot type.

    Args:
        robot_type: Robot type to filter by

    Returns:
        List of factory instances
    """
    return _registry.get_factories_by_type(robot_type)


def search_factories(
    query: str,
    robot_type: Optional[RobotType] = None,
    tags: Optional[list[str]] = None,
    enabled_only: bool = True,
) -> list[str]:
    """
    Search for factories based on various criteria.

    Args:
        query: Search query (matches name or description)
        robot_type: Filter by robot type
        tags: Filter by tags (must have all specified tags)
        enabled_only: Whether to include only enabled factories

    Returns:
        List of matching factory names
    """
    return _registry.search_factories(query, robot_type, tags, enabled_only)


def get_registry() -> RobotFactoryRegistry:
    """
    Get the global factory registry.

    Returns:
        RobotFactoryRegistry: The global registry instance
    """
    return _registry


def get_default_fine_model_cfg(robot_type: RobotType) -> dict[str, Any]:
    """
    Get the default configuration for fine mesh models.

    Args:
        robot_type: Type of robot to get configuration for

    Returns:
        Dictionary with default configuration parameters
    """
    if (
        robot_type == RobotType.MODULAR_LEGS
        or robot_type == RobotType.MODULAR_LEGS.value
        or robot_type == "modular_legs"
    ):
        return {"robot_cfg": ROBOT_CFG_AIR1S, "mesh_dict": MESH_DICT_FINE}
    elif (
        robot_type == RobotType.CUSTOM
        or robot_type == RobotType.CUSTOM.value
        or robot_type == "lego_legs"
    ):
        # For lego_legs: use fine preset (high-fidelity meshes)
        return {"preset": "fine"}
    else:
        # For unknown types, return empty dict to let factory use defaults
        logger.warning(f"No fine model config for robot type: {robot_type}, using factory defaults")
        return {}


def get_default_draft_model_cfg(robot_type: RobotType) -> dict[str, Any]:
    """
    Get the default configuration for draft mesh models.

    Args:
        robot_type: Type of robot to get configuration for

    Returns:
        Dictionary with default configuration parameters
    """
    if (
        robot_type == RobotType.MODULAR_LEGS
        or robot_type == RobotType.MODULAR_LEGS.value
        or robot_type == "modular_legs"
    ):
        return {"robot_cfg": ROBOT_CFG_AIR1S, "mesh_dict": MESH_DICT_DRAFT}
    elif (
        robot_type == RobotType.CUSTOM
        or robot_type == RobotType.CUSTOM.value
        or robot_type == "lego_legs"
    ):
        # For lego_legs: use draft preset (simple primitives for fast simulation)
        return {"preset": "draft"}
    else:
        # For unknown types, return empty dict to let factory use defaults
        logger.warning(f"No draft model config for robot type: {robot_type}, using factory defaults")
        return {}
