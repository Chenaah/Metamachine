"""
Modern command generation and management system.
This module provides a unified, configurable command system that supports:
- Configurable command dimensions and ranges
- Multiple sampling strategies (uniform, discrete, onehot, etc.)
- Automatic resampling at specified intervals
- Support for static commands and dynamic generation

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

from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


class CommandSpec:
    """Specification for a single command dimension."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """Initialize command specification.

        Args:
            name: Name of the command dimension
            config: Configuration dictionary containing:
                - type: Sampling type ('uniform', 'discrete', 'constant')
                - range or choices: Value specification
                - initial_value: Optional initial value
        """
        self.name = name
        self.type = config.get("type", "uniform")
        self.range = config.get("range", [-1.0, 1.0])
        self.choices = config.get("choices", None)
        self.initial_value = config.get("initial_value", 0.0)
        self.weight = config.get("weight", 1.0)  # For weighted sampling

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate command specification configuration."""
        valid_types = ["uniform", "discrete", "constant", "gaussian"]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid command type '{self.type}'. Must be one of {valid_types}"
            )

        if self.type == "discrete" and self.choices is None:
            raise ValueError(
                "Command type 'discrete' requires 'choices' to be specified"
            )

        if self.type == "uniform" and len(self.range) != 2:
            raise ValueError("Command type 'uniform' requires 'range' with [min, max]")

    def sample(self) -> float:
        """Sample a value for this command dimension."""
        if self.type == "uniform":
            return float(np.random.uniform(self.range[0], self.range[1]))
        elif self.type == "discrete":
            return float(np.random.choice(self.choices))
        elif self.type == "constant":
            return float(self.initial_value)
        elif self.type == "gaussian":
            mean = self.range[0] if len(self.range) >= 1 else 0.0
            std = self.range[1] if len(self.range) >= 2 else 1.0
            return float(np.random.normal(mean, std))
        else:
            raise ValueError(f"Unsupported sampling type: {self.type}")


class CommandManager:
    """Modern command generation and management system."""

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize command manager.

        Args:
            cfg: Configuration object containing command specifications
        """
        self.cfg = cfg
        self.task_cfg = cfg.get("task", {})
        self.command_cfg = self.task_cfg.get("commands", {})

        # Check for one-hot mode
        # When enabled, commands form a one-hot vector (only one is 1.0, rest are 0.0)
        self.onehot_mode = self.command_cfg.get("onehot_mode", False)

        # Initialize command specifications
        self._setup_command_specs()

        # Initialize state tracking
        self.step_count = 0
        self.last_resample_step = 0
        # Handle legacy resampling_time setting
        if "resampling_time" in self.task_cfg:
            self.resampling_interval = self.task_cfg.get("resampling_time", 10)
        else:
            self.resampling_interval = self.command_cfg.get("resampling_interval", 10)

        # Initialize command vector
        self.commands = np.zeros(self.num_commands, dtype=object)
        self._sample_all_commands()

    def _setup_command_specs(self) -> None:
        """Setup command specifications from configuration."""
        self.command_specs: list[CommandSpec] = []

        # Handle legacy configuration format
        if "command_x_choices" in self.task_cfg or "commands_ranges" in self.task_cfg:
            self._setup_legacy_commands()
        elif "commands" in self.task_cfg and "dimensions" in self.command_cfg:
            # Modern configuration format
            command_configs = self.command_cfg.get("dimensions", [])

            if isinstance(command_configs, (dict, DictConfig)) and command_configs:
                # Convert dict format to command specs
                for name, config in command_configs.items():
                    # Create a copy to avoid modifying the original config
                    if isinstance(config, DictConfig):
                        config_dict = OmegaConf.to_container(config, resolve=True)
                    else:
                        config_dict = dict(config)

                    # Ensure config_dict is a proper dict
                    if not isinstance(config_dict, dict):
                        config_dict = {}

                    spec = CommandSpec(str(name), config_dict)  # type: ignore
                    self.command_specs.append(spec)
            elif isinstance(command_configs, list) and command_configs:
                # Handle list format
                for i, cmd_config in enumerate(command_configs):
                    if isinstance(cmd_config, dict):
                        name = cmd_config.get("name", f"command_{i}")
                        self.command_specs.append(CommandSpec(name, cmd_config))
                    else:
                        raise ValueError(f"Invalid command config format: {cmd_config}")

        # Default to 3D commands if none specified
        if not self.command_specs:
            self._setup_default_commands()

    def _setup_legacy_commands(self) -> None:
        """Setup commands from legacy configuration format."""
        # Handle legacy command_x_choices
        command_x_choices = self.task_cfg.get("command_x_choices")
        if command_x_choices:
            if isinstance(command_x_choices, list):
                spec = CommandSpec(
                    "x_velocity", {"type": "discrete", "choices": command_x_choices}
                )
                self.command_specs.append(spec)
            elif command_x_choices == "one_hot":
                spec = CommandSpec(
                    "onehot_command",
                    {
                        "type": "discrete",
                        "choices": [0, 1, 2],  # Will be converted to onehot in sampling
                    },
                )
                self.command_specs.append(spec)

        # Handle legacy commands_ranges
        commands_ranges = self.task_cfg.get("commands_ranges")
        if commands_ranges:
            # Handle both regular lists and OmegaConf ListConfig
            if isinstance(commands_ranges, (list, ListConfig)):
                # Convert OmegaConf ListConfig to regular list if needed
                if isinstance(commands_ranges, ListConfig):
                    commands_list = OmegaConf.to_container(
                        commands_ranges, resolve=True
                    )
                else:
                    commands_list = commands_ranges

                # Ensure we have a valid list
                if isinstance(commands_list, list):
                    for i, range_spec in enumerate(commands_list):
                        if (
                            isinstance(range_spec, (list, tuple))
                            and len(range_spec) == 2
                        ):
                            low, high = range_spec
                            spec = CommandSpec(
                                f"command_{i}",
                                {"type": "uniform", "range": [low, high]},
                            )
                            self.command_specs.append(spec)

    def _setup_default_commands(self) -> None:
        """Setup default 3D command specifications."""
        default_specs = [
            {"name": "x_velocity", "type": "uniform", "range": [-1.0, 1.0]},
            {"name": "y_velocity", "type": "uniform", "range": [-1.0, 1.0]},
            {"name": "yaw_rate", "type": "uniform", "range": [-1.0, 1.0]},
        ]

        for spec_config in default_specs:
            self.command_specs.append(CommandSpec(spec_config["name"], spec_config))

    @property
    def num_commands(self) -> int:
        """Number of command dimensions."""
        return len(self.command_specs)

    @property
    def command_names(self) -> list[str]:
        """Names of all command dimensions."""
        return [spec.name for spec in self.command_specs]

    def _sample_all_commands(self) -> None:
        """Sample all command dimensions.
        
        If onehot_mode is enabled, samples a one-hot vector where exactly one
        command is 1.0 and all others are 0.0.
        """
        if self.onehot_mode:
            # One-hot sampling: randomly select one command to be active
            self._sample_onehot_commands()
        else:
            # Standard sampling: each command sampled independently
            for i, spec in enumerate(self.command_specs):
                self.commands[i] = spec.sample()

            # Handle legacy onehot naming convention
            if any("onehot" in spec.name for spec in self.command_specs):
                self._handle_legacy_onehot_commands()

    def _sample_onehot_commands(self) -> None:
        """Sample commands as a one-hot vector.
        
        Randomly selects one command index to be 1.0, all others are 0.0.
        This is used when onehot_mode: true is set in the config.
        """
        # Reset all commands to 0
        self.commands = np.zeros(self.num_commands, dtype=np.float64)
        
        # Randomly select one index to be active
        selected_idx = np.random.randint(self.num_commands)
        self.commands[selected_idx] = 1.0

    def _handle_legacy_onehot_commands(self) -> None:
        """Handle legacy onehot command generation (when 'onehot' is in name)."""
        onehot_specs = [spec for spec in self.command_specs if "onehot" in spec.name]
        if onehot_specs:
            # Convert to onehot encoding
            onehot_commands = np.zeros(len(self.command_specs))
            selected_idx = np.random.randint(len(self.command_specs))
            onehot_commands[selected_idx] = 1.0
            self.commands = onehot_commands

    def step(self) -> None:
        """Update command manager state for one timestep."""
        self.step_count += 1

        # print("Current commands:", self.commands)

        # Check if we need to resample commands
        if self.should_resample():
            self.resample()

    def should_resample(self) -> bool:
        """Check if commands should be resampled."""
        if self.resampling_interval <= 0:
            return False  # No automatic resampling

        return bool(
            (self.step_count - self.last_resample_step) >= self.resampling_interval
        )

    def resample(self) -> None:
        """Resample all commands."""
        self._sample_all_commands()
        self.last_resample_step = self.step_count

    def reset(self) -> None:
        """Reset command manager to initial state."""
        self.step_count = 0
        self.last_resample_step = 0
        self._sample_all_commands()

    def set_command(self, index: int, value: float) -> None:
        """Manually set a command value.

        Args:
            index: Command dimension index
            value: New command value
        """
        if 0 <= index < self.num_commands:
            self.commands[index] = value
        else:
            raise IndexError(
                f"Command index {index} out of range [0, {self.num_commands})"
            )

    def set_command_by_name(self, name: str, value: float) -> None:
        """Manually set a command value by name.

        Args:
            name: Command dimension name
            value: New command value
        """
        try:
            index = self.command_names.index(name)
            self.set_command(index, value)
        except ValueError as e:
            raise ValueError(
                f"Command '{name}' not found. Available: {self.command_names}"
            ) from e

    def set_onehot_by_index(self, active_index: int) -> None:
        """Set commands as a one-hot vector with the specified index active.
        
        Args:
            active_index: Index of the command to set to 1.0 (others become 0.0)
        """
        if not (0 <= active_index < self.num_commands):
            raise IndexError(
                f"Index {active_index} out of range [0, {self.num_commands})"
            )
        
        self.commands = np.zeros(self.num_commands, dtype=np.float64)
        self.commands[active_index] = 1.0

    def set_onehot_by_name(self, active_name: str) -> None:
        """Set commands as a one-hot vector with the named command active.
        
        Args:
            active_name: Name of the command to set to 1.0 (others become 0.0)
        """
        try:
            index = self.command_names.index(active_name)
            self.set_onehot_by_index(index)
        except ValueError as e:
            raise ValueError(
                f"Command '{active_name}' not found. Available: {self.command_names}"
            ) from e

    def get_command_by_name(self, name: str) -> float:
        """Get a command value by name.

        Args:
            name: Command dimension name

        Returns:
            Current value of the named command
        """
        try:
            index = self.command_names.index(name)
            return float(self.commands[index])
        except ValueError as e:
            raise ValueError(
                f"Command '{name}' not found. Available: {self.command_names}"
            ) from e

    def get_command(self, index: int) -> float:
        """Get a command value by index.

        Args:
            index: Command dimension index

        Returns:
            Current value of the command at given index
        """
        if 0 <= index < self.num_commands:
            return float(self.commands[index])
        else:
            raise IndexError(
                f"Command index {index} out of range [0, {self.num_commands})"
            )

    def get_commands_dict(self) -> dict[str, float]:
        """Get all commands as a dictionary mapping names to values.

        Returns:
            Dictionary with command names as keys and current values as values
        """
        return dict(zip(self.command_names, self.commands))

    def get_command_info(self) -> dict[str, Any]:
        """Get comprehensive information about current commands."""
        return {
            "values": self.commands.copy(),
            "names": self.command_names,
            "specs": [
                {
                    "name": spec.name,
                    "type": spec.type,
                    "range": spec.range,
                    "choices": spec.choices,
                }
                for spec in self.command_specs
            ],
            "step_count": self.step_count,
            "last_resample_step": self.last_resample_step,
            "resampling_interval": self.resampling_interval,
        }

    def handle_custom_commands(
        self, command_type: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Handle custom command types for backward compatibility.

        Args:
            command_type: Type of custom command generation

        Returns:
            Tuple of (commands, info_dict)
        """
        info: dict[str, Any] = {}

        if command_type == "onehot_dirichlet":
            # Progressive transition from onehot to dirichlet
            if (
                hasattr(self.cfg, "trainer")
                and self.step_count < self.cfg.trainer.total_steps / 2
            ):
                commands = np.zeros(3)
                commands[np.random.randint(3)] = 1
            else:
                commands = np.random.dirichlet(np.ones(3))
        elif command_type == "onehot":
            commands = np.zeros(3)
            commands[np.random.randint(3)] = 1
        else:
            # Use regular sampling
            commands = self.commands.copy()

        return commands, info
