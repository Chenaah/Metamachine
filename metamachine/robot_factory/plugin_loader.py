"""
Robot Factory Plugin System

This module provides a flexible plugin architecture for robot factories,
enabling:
- Dynamic loading of factory plugins from directories
- Easy separation of private/proprietary factories
- Runtime registration and discovery
- Plugin validation and error handling

The plugin system allows you to:
1. Keep some factories in separate repositories
2. Load factories at runtime without modifying core code
3. Selectively include/exclude factories for different releases

Plugin Structure:
    A plugin is a Python package that contains:
    - __init__.py with a `register_plugin(registry)` function
    - Factory classes implementing BaseRobotFactory
    - Any additional resources (meshes, configs, etc.)

Example plugin __init__.py:
    ```python
    from metamachine.robot_factory import BaseRobotFactory, RobotType
    from .my_factory import MyFactory
    
    def register_plugin(registry):
        registry.register_factory(
            factory_class=MyFactory,
            name="my_factory",
            robot_type=RobotType.CUSTOM,
            description="My custom robot factory",
        )
    ```

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .factory_registry import RobotFactoryRegistry

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""
    pass


class PluginStatus(Enum):
    """Status of a plugin."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """Information about a loaded plugin."""
    name: str
    path: Path
    status: PluginStatus = PluginStatus.NOT_LOADED
    version: str = "unknown"
    description: str = ""
    author: str = ""
    factories_registered: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    module: Optional[Any] = None


class PluginLoader:
    """
    Loads and manages robot factory plugins.
    
    The PluginLoader scans directories for plugins and loads them
    dynamically, registering their factories with the registry.
    
    Example:
        >>> loader = PluginLoader(registry)
        >>> loader.add_plugin_directory("/path/to/plugins")
        >>> loader.load_all_plugins()
        >>> print(loader.list_plugins())
    """
    
    # Standard plugin entry point function name
    REGISTER_FUNCTION = "register_plugin"
    
    # Plugin metadata attributes to look for
    METADATA_ATTRS = ["__plugin_name__", "__version__", "__description__", "__author__"]
    
    def __init__(
        self,
        registry: "RobotFactoryRegistry",
        auto_load_builtin: bool = True,
    ):
        """
        Initialize the PluginLoader.
        
        Args:
            registry: The factory registry to register plugins with
            auto_load_builtin: Whether to auto-load built-in plugins
        """
        self._registry = registry
        self._plugins: dict[str, PluginInfo] = {}
        self._plugin_directories: list[Path] = []
        self._disabled_plugins: set[str] = set()
        
        # Add default plugin directories
        self._add_default_directories()
        
        if auto_load_builtin:
            self._load_builtin_plugins()
    
    def _add_default_directories(self) -> None:
        """Add default plugin search directories."""
        # Current package's plugins directory
        package_dir = Path(__file__).parent
        
        # Built-in plugins (modular_legs)
        builtin_dir = package_dir
        if builtin_dir.exists():
            self._plugin_directories.append(builtin_dir)
        
        # User plugins directory (if exists)
        user_plugins = Path.home() / ".metamachine" / "plugins"
        if user_plugins.exists():
            self._plugin_directories.append(user_plugins)
        
        # Environment variable for additional plugin paths
        env_plugins = os.environ.get("METAMACHINE_PLUGIN_PATH", "")
        if env_plugins:
            for path_str in env_plugins.split(os.pathsep):
                path = Path(path_str)
                if path.exists():
                    self._plugin_directories.append(path)
    
    def _load_builtin_plugins(self) -> None:
        """Load built-in plugins (modular_legs)."""
        # These are loaded automatically by factory_registry
        # This method ensures they're tracked in the plugin system
        # Note: lego_legs is now a separate plugin and must be loaded via
        #       load_plugins_from("/path/to/private_plugins")
        builtin_plugins = [
            ("modular_legs", "ModularLegs factory for sequential morphology robots"),
        ]
        
        for name, description in builtin_plugins:
            plugin_path = Path(__file__).parent / name
            if plugin_path.exists():
                self._plugins[name] = PluginInfo(
                    name=name,
                    path=plugin_path,
                    status=PluginStatus.LOADED,
                    description=description,
                    factories_registered=[],  # Will be filled when actual registration happens
                )
    
    def add_plugin_directory(self, path: str | Path) -> bool:
        """
        Add a directory to search for plugins.
        
        Args:
            path: Directory path
            
        Returns:
            bool: True if directory was added
        """
        path = Path(path)
        if not path.exists():
            logger.warning(f"Plugin directory does not exist: {path}")
            return False
        
        if not path.is_dir():
            logger.warning(f"Plugin path is not a directory: {path}")
            return False
        
        if path not in self._plugin_directories:
            self._plugin_directories.append(path)
            logger.info(f"Added plugin directory: {path}")
            return True
        
        return False
    
    def discover_plugins(self) -> list[str]:
        """
        Discover available plugins in all plugin directories.
        
        Returns:
            List of discovered plugin names
        """
        discovered = []
        
        for directory in self._plugin_directories:
            for item in directory.iterdir():
                # Check if it's a valid plugin package
                if self._is_valid_plugin(item):
                    plugin_name = item.name
                    if plugin_name not in self._plugins:
                        self._plugins[plugin_name] = PluginInfo(
                            name=plugin_name,
                            path=item,
                            status=PluginStatus.NOT_LOADED,
                        )
                        discovered.append(plugin_name)
        
        return discovered
    
    def _is_valid_plugin(self, path: Path) -> bool:
        """
        Check if a path is a valid plugin.
        
        A valid plugin is a directory with an __init__.py that has
        a register_plugin function.
        """
        if not path.is_dir():
            return False
        
        init_file = path / "__init__.py"
        if not init_file.exists():
            return False
        
        # Quick check for register_plugin in file
        try:
            content = init_file.read_text()
            return self.REGISTER_FUNCTION in content or "register_factory" in content
        except Exception:
            return False
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            bool: True if plugin was loaded successfully
        """
        if plugin_name in self._disabled_plugins:
            logger.info(f"Plugin '{plugin_name}' is disabled, skipping")
            return False
        
        if plugin_name not in self._plugins:
            # Try to discover it
            self.discover_plugins()
            if plugin_name not in self._plugins:
                logger.error(f"Plugin '{plugin_name}' not found")
                return False
        
        plugin_info = self._plugins[plugin_name]
        
        if plugin_info.status == PluginStatus.LOADED:
            logger.debug(f"Plugin '{plugin_name}' already loaded")
            return True
        
        plugin_info.status = PluginStatus.LOADING
        
        try:
            # Load the module
            module = self._import_plugin_module(plugin_info.path, plugin_name)
            plugin_info.module = module
            
            # Extract metadata
            self._extract_metadata(plugin_info, module)
            
            # Call the register function if it exists
            if hasattr(module, self.REGISTER_FUNCTION):
                register_func = getattr(module, self.REGISTER_FUNCTION)
                
                # Create a tracking wrapper for the registry
                registered_factories = []
                original_register = self._registry.register_factory
                
                def tracking_register(*args, **kwargs):
                    result = original_register(*args, **kwargs)
                    if result and "name" in kwargs:
                        registered_factories.append(kwargs["name"])
                    elif result and len(args) >= 2:
                        registered_factories.append(args[1])
                    return result
                
                self._registry.register_factory = tracking_register
                try:
                    register_func(self._registry)
                finally:
                    self._registry.register_factory = original_register
                
                plugin_info.factories_registered = registered_factories
            
            plugin_info.status = PluginStatus.LOADED
            logger.info(f"Successfully loaded plugin '{plugin_name}'")
            return True
            
        except Exception as e:
            plugin_info.status = PluginStatus.FAILED
            plugin_info.error_message = str(e)
            logger.error(f"Failed to load plugin '{plugin_name}': {e}")
            return False
    
    def _import_plugin_module(self, path: Path, name: str) -> Any:
        """Import a plugin module from path."""
        # Add parent directory to sys.path if needed
        parent_dir = str(path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        try:
            # Try importing as a package
            spec = importlib.util.spec_from_file_location(
                name,
                path / "__init__.py",
                submodule_search_locations=[str(path)],
            )
            
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not load module spec for {name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            
            return module
            
        except Exception as e:
            raise PluginLoadError(f"Failed to import plugin module: {e}") from e
    
    def _extract_metadata(self, plugin_info: PluginInfo, module: Any) -> None:
        """Extract metadata from a plugin module."""
        if hasattr(module, "__plugin_name__"):
            plugin_info.name = module.__plugin_name__
        if hasattr(module, "__version__"):
            plugin_info.version = module.__version__
        if hasattr(module, "__description__"):
            plugin_info.description = module.__description__
        if hasattr(module, "__author__"):
            plugin_info.author = module.__author__
    
    def load_all_plugins(self) -> dict[str, bool]:
        """
        Load all discovered plugins.
        
        Returns:
            Dict mapping plugin names to success status
        """
        self.discover_plugins()
        
        results = {}
        for plugin_name in self._plugins:
            results[plugin_name] = self.load_plugin(plugin_name)
        
        return results
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Note: This only marks the plugin as not loaded. The factories
        remain registered until the registry is reset.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if plugin was unloaded
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin_info = self._plugins[plugin_name]
        
        # Unregister factories
        for factory_name in plugin_info.factories_registered:
            self._registry.unregister_factory(factory_name)
        
        # Remove module from sys.modules
        if plugin_info.module is not None:
            module_name = plugin_info.module.__name__
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        plugin_info.status = PluginStatus.NOT_LOADED
        plugin_info.module = None
        plugin_info.factories_registered = []
        
        logger.info(f"Unloaded plugin '{plugin_name}'")
        return True
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a disabled plugin."""
        if plugin_name in self._disabled_plugins:
            self._disabled_plugins.remove(plugin_name)
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a plugin (prevents loading).
        
        Args:
            plugin_name: Name of the plugin to disable
            
        Returns:
            bool: True if plugin was disabled
        """
        if plugin_name in self._plugins:
            self._disabled_plugins.add(plugin_name)
            if self._plugins[plugin_name].status == PluginStatus.LOADED:
                self.unload_plugin(plugin_name)
            self._plugins[plugin_name].status = PluginStatus.DISABLED
            return True
        return False
    
    def list_plugins(self) -> list[dict[str, Any]]:
        """
        List all known plugins with their status.
        
        Returns:
            List of plugin information dictionaries
        """
        return [
            {
                "name": info.name,
                "path": str(info.path),
                "status": info.status.value,
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "factories": info.factories_registered,
                "error": info.error_message,
            }
            for info in self._plugins.values()
        ]
    
    def get_plugin_info(self, plugin_name: str) -> Optional[dict[str, Any]]:
        """
        Get information about a specific plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin information dictionary or None
        """
        if plugin_name not in self._plugins:
            return None
        
        info = self._plugins[plugin_name]
        return {
            "name": info.name,
            "path": str(info.path),
            "status": info.status.value,
            "version": info.version,
            "description": info.description,
            "author": info.author,
            "factories": info.factories_registered,
            "error": info.error_message,
        }
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            bool: True if plugin was reloaded successfully
        """
        if plugin_name in self._plugins:
            self.unload_plugin(plugin_name)
        return self.load_plugin(plugin_name)


# Singleton instance for convenience
_plugin_loader: Optional[PluginLoader] = None


def get_plugin_loader(registry: Optional["RobotFactoryRegistry"] = None) -> PluginLoader:
    """
    Get the global plugin loader instance.
    
    Args:
        registry: Optional registry to use (only needed for first call)
        
    Returns:
        The global PluginLoader instance
    """
    global _plugin_loader
    
    if _plugin_loader is None:
        if registry is None:
            from .factory_registry import get_registry
            registry = get_registry()
        _plugin_loader = PluginLoader(registry)
    
    return _plugin_loader


def load_plugins_from_directory(path: str | Path) -> dict[str, bool]:
    """
    Convenience function to load plugins from a directory.
    
    Args:
        path: Directory containing plugins
        
    Returns:
        Dict mapping plugin names to success status
    """
    loader = get_plugin_loader()
    loader.add_plugin_directory(path)
    return loader.load_all_plugins()

