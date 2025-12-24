"""
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
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces.box import Box
from omegaconf import OmegaConf

from ..utils.kbhit import KBHit
from ..utils.validation import is_number
from .components.action import ActionProcessor
from .components.command import CommandManager
from .components.reward import RewardCalculator
from .components.state import State
from .components.termination import TerminationChecker
from .components.visualization import Visualizer


class Base(gym.Env, ABC):
    """Modern base environment for robotics control.

    This class provides a unified, component-based interface for both simulation
    and real robot environments. It integrates all modern components for state
    management, action processing, reward calculation, and termination checking.
    """

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize the environment with modern component architecture.

        Args:
            cfg: Modern configuration object with standardized structure
        """
        self.cfg = self._validate_config(cfg)

        # Initialize all components
        self._initialize_components()

        # Setup gym interface
        self._setup_spaces()

        # Initialize state tracking
        self._initialize_state()

        # Setup control interface
        self._setup_control()

        if not hasattr(self, "_log_dir"):
            self._setup_logging()
        #

    def _validate_config(self, cfg: OmegaConf) -> OmegaConf:
        """Validate configuration has required modern structure."""
        required_sections = ["environment", "control", "observation", "task"]
        missing = [s for s in required_sections if s not in cfg]
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        return cfg

    def _initialize_components(self) -> None:
        """Initialize all modern environment components."""
        self.state = State(self.cfg)
        self.action_processor = ActionProcessor(self.cfg)
        self.reward_calculator = RewardCalculator(self.cfg)
        self.termination_checker = TerminationChecker(self.cfg)
        self.command_manager = CommandManager(self.cfg)

        # Link command manager to state for named access
        self.state.set_command_manager(self.command_manager)

        # Optional visualization
        if self.cfg.get("logging", {}).get("enable_visualization", False):
            self.visualizer = Visualizer(self.cfg)
        else:
            self.visualizer = None

    def _setup_spaces(self) -> None:
        """Setup gym action and observation spaces."""
        # Action space (adjusted for frozen joints)
        action_limit = self.cfg.control.symmetric_limit
        effective_action_size = self.action_processor.effective_action_size
        self.action_space = Box(
            low=-action_limit,
            high=action_limit,
            shape=(effective_action_size,),
            dtype=np.float32,
        )

        # Observation space
        obs_size = self.state.num_obs * self.cfg.observation.include_history_steps
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _setup_control(self) -> None:
        """Setup control interface."""
        self.kb = KBHit()
        self.input_key = ""
        self.policy_switch = 0

        # Action tracking (adjusted for frozen joints)
        total_joints = self.cfg.control.num_actions
        self.last_action_flat = np.zeros(total_joints * self.cfg.environment.num_envs)
        self.last_reward = 0

    def _initialize_state(self) -> None:
        """Initialize episode state tracking."""
        self.step_count = 0
        self.episode_rewards = []
        self.t0 = None

    @property
    def commands(self) -> np.ndarray:
        """Get current command vector."""
        return self.command_manager.commands

    @property
    def dt(self) -> float:
        """Environment timestep."""
        return self.cfg.control.dt

    @abstractmethod
    def _reset_robot(self) -> None:
        """Reset robot to initial state."""
        pass

    @abstractmethod
    def _get_observable_data(self) -> dict[str, Any]:
        """Get current observable state data."""
        pass

    def _update_state(self) -> None:
        """Update state with current observations and commands."""
        obs_data = self._get_observable_data()
        obs_data.update(
            {
                "last_action": self.last_action_flat,
                "last_reward": self.last_reward,
                # "commands": self.commands
            }
        )
        self.state.update(obs_data)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        # Reset all components
        self._reset_robot()
        self.action_processor.reset(self.state)
        self.state.reset()
        self.reward_calculator.reset()
        self.command_manager.reset()

        # Reset filters
        # if self.action_filter:
        #     self.action_filter.reset(init_hist=self.state.dof_pos)
        # if self.action_melter:
        #     self.action_melter.reset()

        # Reset state tracking
        self._initialize_state()
        self.termination_checker.reset()

        # Resample commands
        self._resample_commands()

        # Get initial observation
        self._update_state()
        obs = self.state.get_observation(reset=True)

        self.t0 = time.time()
        return obs, {}

    def _resample_commands(self):
        """Resample command targets using the modern command manager."""
        self.command_manager.resample()

    @abstractmethod
    def _perform_action(self, action: np.ndarray) -> dict[str, Any]:
        """Execute action on robot.

        Args:
            action: Processed action to execute

        Returns:
            Action execution info
        """
        pass

    def _check_termination(self) -> tuple[bool, bool]:
        """Check termination conditions.

        Returns:
            (done, truncated)
        """
        done = self.termination_checker.check_done(self.state)
        truncated = self.termination_checker.check_truncated(self.state)
        return done, truncated

    def _handle_input(self):
        """Handle keyboard input for manual control."""
        if self.kb.kbhit():
            self.input_key = self.kb.getch()
            if is_number(self.input_key):
                self.policy_switch = int(self.input_key)
            elif self.input_key == "j":
                # Jump command - set using command manager
                if "x_velocity" in self.command_manager.command_names:
                    self.command_manager.set_command_by_name("x_velocity", 1.0)
                elif len(self.command_manager.commands) > 0:
                    self.command_manager.set_command(0, 1.0)

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: Raw action from policy

        Returns:
            (observation, reward, done, truncated, info)
        """
        # Process action
        processed_action = self.action_processor.process(action)
        self.last_action_flat = self.action_processor.last_action_flat

        # Execute action
        action_info = self._perform_action(processed_action)

        # Update state
        self._update_state()

        # Calculate reward
        obs = self.state.get_observation()
        reward, reward_info = self.reward_calculator.calculate(self.state)
        self.last_reward = reward

        # Check termination
        done, truncated = self._check_termination()

        # Update tracking
        self.step_count += 1
        # print("Number of steps:", self.step_count)
        self.episode_rewards.append(reward)
        self.termination_checker.step()

        # Update command manager
        self.command_manager.step()

        # Update visualization
        if self.visualizer:
            self.visualizer.update(self.state)

        # Handle manual input
        self._handle_input()

        # Prepare reward components for info
        reward_components = {}
        if "component_values" in reward_info and "component_weights" in reward_info:
            component_values = reward_info["component_values"]
            component_weights = reward_info["component_weights"]

            for component_name in component_values:
                # Calculate weighted contribution (actual reward contribution)
                weighted_value = (
                    component_values[component_name] * component_weights[component_name]
                )
                reward_components[component_name] = round(weighted_value, 4)

        # Prepare info
        info = {
            "episode_step": self.step_count,
            "total_reward": sum(self.episode_rewards),
            "is_upsidedown": self.termination_checker.check_upsidedown(self.state),
            "reward_components": reward_components,
            **action_info,
            **reward_info,
        }

        if done or truncated:
            self._post_done()

        return obs, reward, done, truncated, info

    def render(self):
        """Render the environment."""
        pass

    def _post_done(self):
        """Handle post-termination cleanup."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "kb"):
            self.kb.set_normal_term()

    def _setup_logging(self) -> None:
        """Setup logging directories and files."""
        self._log_dir = None
        if self.cfg.get("logging", {}).get("create_log_dir", False):
            base_log_dir = self.cfg.logging.get("data_dir", "./logs")
            base_log_dir = "./logs" if base_log_dir is None else base_log_dir
            exp_name = self.cfg.logging.get("experiment_name", None)
            self._log_dir = self.create_log_directory(base_log_dir, exp_name)
        else:
            self._log_dir = self.cfg.logging.get("data_dir", None)

    def create_log_directory(self, log_dir, exp_name: Optional[str] = None):
        """Create a log directory for saving logs with date and component information.
        
        If exp_name is provided and contains a timestamp pattern (like "name-YYYYMMDD-HHMMSS"),
        it will be used directly as the directory name. Otherwise, a directory name will be
        generated with date and component info.
        """
        # If exp_name looks like a complete experiment name (contains date pattern), use it directly
        if exp_name and "-" in exp_name and any(c.isdigit() for c in exp_name):
            # exp_name is like "lego_tripod-20231223-222348", use directly
            dir_name = exp_name
        else:
            # Generate directory name with date and component info (legacy behavior)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Get observation component initials
            obs_initials = ""
            if hasattr(self, "state") and hasattr(self.state, "observation_components"):
                obs_initials = "".join(
                    [comp.name[0] for comp in self.state.observation_components]
                )

            # Get reward component initials
            reward_initials = ""
            if hasattr(self, "reward_calculator") and hasattr(
                self.reward_calculator, "components"
            ):
                reward_initials = "".join(
                    [comp.name[0] for comp in self.reward_calculator.components]
                )

            robot_name = self.cfg.morphology.get("robot_type", "robot")
            robot_name_initials = robot_name[0]
            # Create directory name with date and component info
            dir_components = [date_str + robot_name_initials]
            if obs_initials:
                dir_components.append(f"obs_{obs_initials}")
            if reward_initials:
                dir_components.append(f"rew_{reward_initials}")
            if exp_name:
                dir_components.append(exp_name)

            dir_name = "_".join(dir_components)
        
        self._log_dir = os.path.join(log_dir, dir_name)

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        # Save configuration file
        config_path = os.path.join(self._log_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(self.cfg, f)

        return self._log_dir
