"""
Modular legs robot factory implementation.
This module provides the ModularLegsFactory class which creates modular leg robots
by wrapping the existing ModularLegs implementation with the new factory interface.

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

import logging
from typing import Any, Optional

from ..base_factory import BaseRobot, BaseRobotFactory, RobotType
from .constants import MESH_DICT_FINE, ROBOT_CFG_AIR1S
from .meta_designer import ModularLegs as LegacyModularLegs
from .morphology import ModuleConnection, RobotMorphology

logger = logging.getLogger(__name__)


class ModularLegsRobot(BaseRobot):
    """
    Modular legs robot implementation.

    This class wraps the existing ModularLegs class to provide the standardized
    BaseRobot interface while maintaining compatibility with existing code.
    """

    def __init__(
        self,
        factory: "ModularLegsFactory",
        morphology: Optional[RobotMorphology] = None,
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the modular legs robot.

        Args:
            factory: The factory that created this robot
            morphology: Robot morphology specification
            config: Configuration parameters
            **kwargs: Additional robot-specific parameters
        """
        super().__init__(factory, morphology, config, **kwargs)

    def _initialize_robot(self, **kwargs) -> None:
        """Initialize robot-specific attributes and components."""
        # Extract parameters for ModularLegs
        robot_cfg = self.config.get("robot_cfg", ROBOT_CFG_AIR1S)
        mesh_dict = self.config.get("mesh_dict", MESH_DICT_FINE)
        broken_mask = self.config.get("broken_mask", None)
        sim_cfg = self.config.get("sim_cfg", None)
        log_dir = self.config.get("log_dir", kwargs.get("log_dir", None))

        # Create the underlying ModularLegs instance
        self._modular_legs = LegacyModularLegs(
            morphology=self.morphology,
            robot_cfg=robot_cfg,
            mesh_dict=mesh_dict,
            broken_mask=broken_mask,
            sim_cfg=sim_cfg,
            log_dir=log_dir,
        )

        # Track additional state
        self._properties_cache = {}
        self._is_valid_cache = None

    def reset(self, **kwargs) -> None:
        """Reset the robot to its initial state."""
        init_a_module = kwargs.get("init_a_module", True)
        self._modular_legs.reset(init_a_module=init_a_module)
        self._clear_cache()

    def add_module(self, module_spec: ModuleConnection) -> bool:
        """
        Add a module to the robot.

        Args:
            module_spec: Module connection specification

        Returns:
            bool: True if module was added successfully, False otherwise
        """
        try:
            self._modular_legs.add_module(module_spec)
            self._clear_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to add module: {e}")
            return False

    def remove_module(self, module_id: Any) -> bool:
        """
        Remove a module from the robot.

        Args:
            module_id: Module identifier

        Returns:
            bool: True if module was removed successfully, False otherwise
        """
        # Note: The original ModularLegs doesn't have a remove_module method
        # This would need to be implemented if needed
        logger.warning("Remove module not implemented for ModularLegs")
        return False

    def get_available_connections(self) -> list[Any]:
        """
        Get available connection points for new modules.

        Returns:
            List of available connection specifications
        """
        try:
            available_modules = self._modular_legs.get_available_module_ids()
            connections = []

            for module_id in available_modules:
                available_docks = self._modular_legs.get_available_docks(module_id)
                for dock in available_docks:
                    connections.append(
                        {
                            "module_id": module_id,
                            "dock": dock,
                            "available_rotations": self._modular_legs.get_available_rotation_ids(
                                module_id, dock.value
                            ),
                        }
                    )

            return connections
        except Exception as e:
            logger.error(f"Failed to get available connections: {e}")
            return []

    def save(self, save_dir: str, **kwargs) -> str:
        """
        Save the robot to files.

        Args:
            save_dir: Directory to save files
            **kwargs: Additional save parameters

        Returns:
            str: Path to the saved main file
        """
        try:
            return self._modular_legs.save(save_dir, **kwargs)
        except Exception as e:
            logger.error(f"Failed to save robot: {e}")
            raise

    def get_xml_string(self) -> str:
        """
        Get the robot as an XML string.

        Returns:
            str: XML representation of the robot
        """
        try:
            return self._modular_legs.get_string()
        except Exception as e:
            logger.error(f"Failed to get XML string: {e}")
            raise

    def render(self, **kwargs):
        """
        Render the robot for visualization.

        Args:
            **kwargs: Rendering parameters
        """
        try:
            fix = kwargs.get("fix", False)
            pos = kwargs.get("pos", (0, 0, 0.4))
            quat = kwargs.get("quat", [1, 0, 0, 0])
            joint_pos = kwargs.get("joint_pos", None)

            self._modular_legs.render(fix=fix, pos=pos, quat=quat, joint_pos=joint_pos)
        except Exception as e:
            logger.error(f"Failed to render robot: {e}")
            raise

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the robot configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if self._is_valid_cache is not None:
            return self._is_valid_cache

        errors = []

        try:
            # Check if robot has at least one module
            if (
                not hasattr(self._modular_legs, "module_id_counter")
                or self._modular_legs.module_id_counter == 0
            ):
                errors.append("Robot has no modules")

            # Check if morphology is valid
            if self.morphology is not None and not isinstance(
                self.morphology, RobotMorphology
            ):
                errors.append("Invalid morphology type")

            # Additional validation could be added here
            # For example, checking joint limits, mass distribution, etc.

            is_valid = len(errors) == 0
            self._is_valid_cache = (is_valid, errors)
            return is_valid, errors

        except Exception as e:
            errors.append(f"Validation error: {e}")
            self._is_valid_cache = (False, errors)
            return False, errors

    def get_properties(self) -> dict[str, Any]:
        """
        Get robot properties and statistics.

        Returns:
            Dictionary of robot properties
        """
        if self._properties_cache:
            return self._properties_cache.copy()

        try:
            properties = {
                "robot_type": self.factory.robot_type.value,
                "num_modules": getattr(self._modular_legs, "module_id_counter", 0),
                "config": self.config.copy(),
                "morphology_type": (
                    type(self.morphology).__name__ if self.morphology else None
                ),
            }

            # Add designer properties if available
            if hasattr(self._modular_legs, "designer") and hasattr(
                self._modular_legs.designer, "robot_properties"
            ):
                properties.update(self._modular_legs.designer.robot_properties)

            self._properties_cache = properties
            return properties.copy()

        except Exception as e:
            logger.error(f"Failed to get properties: {e}")
            return {"robot_type": self.factory.robot_type.value, "error": str(e)}

    def _clear_cache(self):
        """Clear cached data."""
        self._properties_cache.clear()
        self._is_valid_cache = None

    def paint(self, color: str = "black"):
        """
        Paint the robot with a specific color.

        Args:
            color: Color name
        """
        try:
            self._modular_legs.paint(color)
        except Exception as e:
            logger.error(f"Failed to paint robot: {e}")
            raise

    def get_available_module_ids(self) -> list[int]:
        """Get available module IDs."""
        try:
            return self._modular_legs.get_available_module_ids()
        except Exception as e:
            logger.error(f"Failed to get available module IDs: {e}")
            return []

    def get_available_docks(self, module_id: int) -> list[Any]:
        """Get available docks for a module."""
        try:
            return self._modular_legs.get_available_docks(module_id)
        except Exception as e:
            logger.error(f"Failed to get available docks: {e}")
            return []


class ModularLegsFactory(BaseRobotFactory):
    """
    Factory for creating modular legs robots.

    This factory creates robots with modular leg components that can be
    connected in various configurations for different locomotion patterns.
    """

    def __init__(
        self,
        robot_cfg: Optional[dict[str, Any]] = None,
        mesh_dict: Optional[dict[str, str]] = None,
        sim_cfg: Optional[dict[str, Any]] = None,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the modular legs factory.

        Args:
            robot_cfg: Robot configuration parameters
            mesh_dict: Mesh dictionary for component rendering
            sim_cfg: Simulation configuration parameters
            log_dir: Directory for logging and saving outputs
            **kwargs: Additional factory parameters
        """
        super().__init__(
            robot_type=RobotType.MODULAR_LEGS,
            name="ModularLegs",
            description="Factory for creating modular legs robots with configurable morphologies",
            default_config={
                "robot_cfg": robot_cfg or ROBOT_CFG_AIR1S,
                "mesh_dict": mesh_dict or MESH_DICT_FINE,
                "sim_cfg": sim_cfg,
                "log_dir": log_dir,
            },
            supported_morphologies=["RobotMorphology", "list", "sequence"],
            **kwargs,
        )

    def _initialize_factory(self, **kwargs) -> None:
        """Initialize factory-specific attributes and components."""
        # Store factory-specific configuration
        self._default_robot_cfg = self.default_config.get("robot_cfg", ROBOT_CFG_AIR1S)
        self._default_mesh_dict = self.default_config.get("mesh_dict", MESH_DICT_FINE)

        # Initialize any factory-specific components
        self._capabilities = [
            "modular_design",
            "dynamic_morphology",
            "xml_export",
            "mujoco_simulation",
            "visual_rendering",
            "joint_control",
            "sensor_integration",
        ]

    def create_robot(
        self,
        morphology: Optional[RobotMorphology] = None,
        config: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> ModularLegsRobot:
        """
        Create a new modular legs robot instance.

        Args:
            morphology: Robot morphology specification
            config: Configuration parameters
            **kwargs: Additional creation parameters

        Returns:
            ModularLegsRobot: A robot instance
        """
        # Merge configurations
        final_config = self.merge_config(self.default_config, config or {})

        # Validate morphology if provided
        if morphology is not None and not self.validate_morphology(morphology):
            raise ValueError(f"Invalid morphology for {self.name}")

        # Create the robot
        robot = ModularLegsRobot(
            factory=self, morphology=morphology, config=final_config, **kwargs
        )

        return robot

    def get_available_configurations(self) -> list[dict[str, Any]]:
        """
        Get list of available robot configurations.

        Returns:
            List of configuration dictionaries
        """
        return [
            {
                "name": "standard",
                "description": "Standard modular legs configuration",
                "robot_cfg": ROBOT_CFG_AIR1S,
                "mesh_dict": MESH_DICT_FINE,
            },
            {
                "name": "hybrid",
                "description": "Hybrid configuration with simplified geometry",
                "robot_cfg": self._default_robot_cfg,
                "mesh_dict": self._default_mesh_dict,
            },
        ]

    def validate_morphology(self, morphology: Any) -> bool:
        """
        Validate if a morphology is supported by this factory.

        Args:
            morphology: The morphology to validate

        Returns:
            bool: True if morphology is valid, False otherwise
        """
        try:
            # Check if it's a RobotMorphology instance
            if isinstance(morphology, RobotMorphology):
                return True

            # Check if it's a list-like sequence that can be converted
            if hasattr(morphology, "__iter__") and not isinstance(
                morphology, (str, bytes)
            ):
                return True

            # Check if it's None (will use default)
            if morphology is None:
                return True

            return False

        except Exception as e:
            logger.error(f"Error validating morphology: {e}")
            return False

    def get_default_morphology(self) -> RobotMorphology:
        """
        Get the default morphology for this robot type.

        Returns:
            Default morphology specification
        """
        return RobotMorphology()

    def get_capabilities(self) -> list[str]:
        """
        Get the capabilities supported by this factory.

        Returns:
            List of capability strings
        """
        return self._capabilities.copy()

    def create_basic_quadruped(self, **kwargs) -> ModularLegsRobot:
        """
        Create a basic quadruped robot configuration.

        Args:
            **kwargs: Additional creation parameters

        Returns:
            ModularLegsRobot: A quadruped robot instance
        """
        # Create a morphology for a basic quadruped
        morphology = RobotMorphology()

        # Add basic quadruped connections
        from .morphology import DockPosition

        connections = [
            ModuleConnection(
                parent_module_id=0,
                parent_dock=DockPosition.UPPER_STICK_2,
                child_dock=DockPosition.UPPER_STICK_1,
                orientation=0,
            ),
            ModuleConnection(
                parent_module_id=0,
                parent_dock=DockPosition.UPPER_STICK_4,
                child_dock=DockPosition.UPPER_STICK_1,
                orientation=0,
            ),
            ModuleConnection(
                parent_module_id=0,
                parent_dock=DockPosition.LOWER_STICK_4,
                child_dock=DockPosition.UPPER_STICK_1,
                orientation=0,
            ),
            ModuleConnection(
                parent_module_id=0,
                parent_dock=DockPosition.LOWER_STICK_6,
                child_dock=DockPosition.UPPER_STICK_1,
                orientation=0,
            ),
        ]

        for connection in connections:
            morphology.add_connection(connection)

        return self.create_robot(morphology=morphology, **kwargs)

    def create_from_legacy_pipeline(
        self, pipeline: list[int], **kwargs
    ) -> ModularLegsRobot:
        """
        Create a robot from a legacy pipeline specification.

        Args:
            pipeline: Legacy pipeline specification
            **kwargs: Additional creation parameters

        Returns:
            ModularLegsRobot: A robot instance
        """
        # Convert legacy pipeline to morphology
        morphology = RobotMorphology.from_sequence(pipeline)

        return self.create_robot(morphology=morphology, **kwargs)
