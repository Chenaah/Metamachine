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

from dataclasses import dataclass, field
import pdb
from typing import Any, Callable, Optional

import numpy as np
from omegaconf import OmegaConf

from ...utils.math_utils import AverageFilter, quat_apply, quat_rotate_inverse


@dataclass
class RawState:
    """Raw state values received from environment."""

    # Position and orientation
    pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    quats: list[np.ndarray] = field(default_factory=list)

    # Velocities
    vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Joint state - size will be set after initialization
    dof_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))  # Will be resized
    dof_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))  # Will be resized

    # Sensor data
    gyros: Optional[np.ndarray] = None
    accs: Optional[np.ndarray] = None

    # Contact information
    contact_floor_balls: list[int] = field(default_factory=list)
    contact_floor_geoms: list[int] = field(default_factory=list)
    contact_floor_socks: list[int] = field(default_factory=list)

    # Goal distance (from distance sensor)
    goal_distance: float = -1.0  # Global goal distance (-1 = no reading)
    goal_distances: np.ndarray = field(default_factory=lambda: np.zeros(1))  # Per-module

    def __init__(self, num_dof: int = 1) -> None:
        """Initialize RawState with specified number of degrees of freedom."""
        # Position and orientation
        self.pos_world = np.zeros(3)
        self.quat = np.zeros(4)
        self.quats = []
        self.goal_distance = -1.0

        # Velocities
        self.vel_body = np.zeros(3)
        self.vel_world = np.zeros(3)
        self.ang_vel_body = np.zeros(3)
        self.ang_vel_world = np.zeros(3)

        # Per-module state
        self.dof_pos = np.zeros(num_dof)
        self.dof_vel = np.zeros(num_dof)
        self.gyros = np.zeros((num_dof, 3))
        self.accs = np.zeros((num_dof, 3))
        self.goal_distances = np.ones(num_dof)*-1.0

        # Contact information
        self.contact_floor_balls = []
        self.contact_floor_geoms = []
        self.contact_floor_socks = []

        # Store initial shapes for validation
        self.__post_init__()

    def __post_init__(self) -> None:
        """Store initial shapes for validation."""
        self._initial_shapes = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                self._initial_shapes[key] = value.shape
            elif isinstance(value, list):
                self._initial_shapes[key] = (len(value),)

    def update(self, data: dict[str, Any]) -> None:
        """Update raw state with new data."""
        for key, value in data.items():
            if hasattr(self, key):
                # Skip validation for None values and list-type fields
                if value is None or key in [
                    "contact_floor_balls",
                    "contact_floor_geoms",
                    "contact_floor_socks",
                    "quats",
                ]:
                    setattr(self, key, value)
                    continue

                # For numpy arrays, validate shape
                if isinstance(value, np.ndarray):
                    expected_shape = self._initial_shapes.get(key)
                    if expected_shape is not None and value.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: expected {expected_shape}, got {value.shape}"
                        )

                setattr(self, key, value)


@dataclass
class AccurateState:
    """Accurate state values for reward computation (simulation only)."""

    quat: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(4))
    vel_body: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))
    vel_world: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))
    pos_world: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))
    ang_vel_body: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(3))

    def update(self, data: dict[str, Any]) -> None:
        """Update accurate state with new data."""
        for key, value in data.items():
            # Only update if the key exists and starts with 'accurate_'
            if key.startswith("accurate_"):
                attr_name = key.replace("accurate_", "")
                if hasattr(self, attr_name):
                    setattr(self, attr_name, value)

    def is_available(self) -> bool:
        """Check if accurate state data is available."""
        return any(
            getattr(self, attr) is not None
            for attr in ["quat", "vel_body", "vel_world", "pos_world", "ang_vel_body"]
        )

    def reset(self) -> None:
        """Reset all accurate state values to None."""
        self.quat = None
        self.vel_body = None
        self.vel_world = None
        self.pos_world = None
        self.ang_vel_body = None


@dataclass
class DerivedState:
    """Derived state values computed from raw state."""

    height: np.ndarray = field(default_factory=lambda: np.zeros(1))
    projected_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_gravities: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    heading: np.ndarray = field(default_factory=lambda: np.zeros(1))
    speed: np.ndarray = field(default_factory=lambda: np.zeros(1))

    def __init__(self, num_dof: int = 1) -> None:
        """Initialize DerivedState with specified number of degrees of freedom."""
        self.projected_gravities = np.zeros((num_dof, 3))
        self.projected_gravity = np.zeros(3)

    def __post_init__(self) -> None:
        """Store initial shapes for validation."""
        self._initial_shapes = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                self._initial_shapes[key] = value.shape
            elif isinstance(value, list):
                self._initial_shapes[key] = (len(value),)

    def update(self, data: dict[str, Any]) -> None:
        """Update derived state with new data."""
        for key, value in data.items():
            if hasattr(self, key):
                # Skip validation for None values and list fields
                if value is None or key in ["projected_gravities"]:
                    setattr(self, key, value)
                    continue

                # For numpy arrays, validate shape
                if isinstance(value, np.ndarray):
                    expected_shape = self._initial_shapes.get(key)
                    if expected_shape is not None and value.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {key}: expected {expected_shape}, got {value.shape}"
                        )

                setattr(self, key, value)


class ObservationComponent:
    """A component that can be included in the observation vector."""

    def __init__(
        self, name: str, data_fn: Callable, transform_fn: Optional[Callable] = None
    ):
        """Initialize observation component.

        Args:
            name: Name of the component
            data_fn: Function that returns the component data
            transform_fn: Optional function to transform the data before including in observation
        """
        self.name = name
        self.data_fn = data_fn
        self.transform_fn = transform_fn if transform_fn else lambda x: x

    def get_data(self, state) -> np.ndarray:
        """Get the component data and apply any transformations."""
        data = self.data_fn(state)
        return self.transform_fn(data)


class ModularObservationComponent:
    """A component for modular observations that accesses indexed data per module.
    
    This component allows building observations where each module gets its own
    slice of the data, enabling patterns like:
    
        obs = []
        for i in range(N_MODULES):
            module_obs = np.concatenate([
                s.projected_gravities[i],
                s.gyros[i],
                s.dof_pos[i:i+1],
                s.dof_vel[i:i+1]
            ])
            obs.append(module_obs)
        obs = np.concatenate(obs)
    """

    def __init__(
        self,
        name: str,
        data_fn: Callable,
        index_type: str = "element",
        slice_size: int = 1,
        transform_fn: Optional[Callable] = None
    ):
        """Initialize modular observation component.

        Args:
            name: Name of the component
            data_fn: Function that returns the component data (list or array)
            index_type: How to index the data:
                - "element": Use data[i] (for lists of arrays like projected_gravities)
                - "slice": Use data[i*size:(i+1)*size] (for flat arrays like dof_pos)
            slice_size: Size of each slice when index_type is "slice" (default: 1)
            transform_fn: Optional function to transform the data
        """
        self.name = name
        self.data_fn = data_fn
        self.index_type = index_type
        self.slice_size = slice_size
        self.transform_fn = transform_fn if transform_fn else lambda x: x

    def get_data_for_module(self, state, module_idx: int) -> np.ndarray:
        """Get the component data for a specific module index.
        
        Args:
            state: The State object
            module_idx: Index of the module (0-indexed)
            
        Returns:
            numpy.ndarray: Data for the specified module
        """
        data = self.data_fn(state)
        
        if self.index_type == "element":
            # Access as list element: data[i]
            try:
                indexed_data = data[module_idx]
            except IndexError:
                import pdb; pdb.set_trace()
        elif self.index_type == "slice":
            # Access as array slice: data[i*size:(i+1)*size]
            start_idx = module_idx * self.slice_size
            end_idx = start_idx + self.slice_size
            indexed_data = data[start_idx:end_idx]
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")
        
        # Ensure numpy array
        if not isinstance(indexed_data, np.ndarray):
            indexed_data = np.array(indexed_data)
            
        return self.transform_fn(indexed_data)


class ActionHistoryBuffer:
    """Handles action history for environments that need multiple timesteps of past actions."""

    def __init__(self, num_actions: int, history_steps: int = 3) -> None:
        """Initialize action history buffer.

        Args:
            num_actions: Number of actions per timestep
            history_steps: Number of timesteps to store in history
        """
        self.num_actions = num_actions
        self.history_steps = history_steps
        self.action_history = np.zeros((history_steps, num_actions))

    def reset(self, initial_action=None) -> None:
        """Reset buffer with initial action (or zeros)."""
        if initial_action is None:
            initial_action = np.zeros(self.num_actions)
        self.action_history = np.tile(initial_action, (self.history_steps, 1))

    def update(self, new_action) -> None:
        """Update history with new action, shifting older actions."""
        # Shift history: [t-2, t-1, t-0] -> [t-1, t-0, new]
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = new_action

    def get_action(self, steps_back: int = 0):
        """Get action from history.

        Args:
            steps_back: How many steps back (0 = most recent, 1 = last_action, 2 = last_last_action)

        Returns:
            numpy.ndarray: Action at specified timestep
        """
        if steps_back >= self.history_steps:
            raise ValueError(
                f"Cannot access {steps_back} steps back, only {self.history_steps} steps stored"
            )
        return self.action_history[-(steps_back + 1)]

    @property
    def last_action(self):
        """Get most recent action (equivalent to get_action(0))."""
        return self.get_action(0)

    @property
    def last_last_action(self):
        """Get previous action (equivalent to get_action(1))."""
        return self.get_action(1)

    @property
    def last_last_last_action(self):
        """Get action from 2 steps ago (equivalent to get_action(2))."""
        return self.get_action(2)

    def get_history_vector(self, num_steps: int = None):
        """Get flattened history vector for observations.

        Args:
            num_steps: Number of recent steps to include (default: all)

        Returns:
            numpy.ndarray: Flattened action history
        """
        if num_steps is None:
            num_steps = self.history_steps

        if num_steps > self.history_steps:
            raise ValueError(
                f"Cannot get {num_steps} steps, only {self.history_steps} stored"
            )

        # Return most recent num_steps actions, flattened
        return self.action_history[-num_steps:].flatten()


class RewardHistoryBuffer:
    """Handles reward history for environments that need multiple timesteps of past rewards."""

    def __init__(self, history_steps: int = 3) -> None:
        """Initialize reward history buffer.

        Args:
            history_steps: Number of timesteps to store in history
        """
        self.history_steps = history_steps
        self.reward_history = np.zeros(history_steps)

    def reset(self, initial_reward=None) -> None:
        """Reset buffer with initial reward (or zeros)."""
        if initial_reward is None:
            initial_reward = 0.0
        self.reward_history = np.full(self.history_steps, initial_reward)

    def update(self, new_reward) -> None:
        """Update history with new reward, shifting older rewards."""
        # Shift history: [t-2, t-1, t-0] -> [t-1, t-0, new]
        self.reward_history[:-1] = self.reward_history[1:]
        self.reward_history[-1] = new_reward

    def get_reward(self, steps_back: int = 0):
        """Get reward from history.

        Args:
            steps_back: How many steps back (0 = most recent, 1 = last_reward, 2 = last_last_reward)

        Returns:
            float: Reward at specified timestep
        """
        if steps_back >= self.history_steps:
            raise ValueError(
                f"Cannot access {steps_back} steps back, only {self.history_steps} steps stored"
            )
        return self.reward_history[-(steps_back + 1)]

    @property
    def last_reward(self):
        """Get most recent reward (equivalent to get_reward(0))."""
        return self.get_reward(0)

    @property
    def last_last_reward(self):
        """Get previous reward (equivalent to get_reward(1))."""
        return self.get_reward(1)

    @property
    def last_last_last_reward(self):
        """Get reward from 2 steps ago (equivalent to get_reward(2))."""
        return self.get_reward(2)

    def get_history_vector(self, num_steps: int = None):
        """Get reward history vector for observations.

        Args:
            num_steps: Number of recent steps to include (default: all)

        Returns:
            numpy.ndarray: Reward history
        """
        if num_steps is None:
            num_steps = self.history_steps

        if num_steps > self.history_steps:
            raise ValueError(
                f"Cannot get {num_steps} steps, only {self.history_steps} stored"
            )

        # Return most recent num_steps rewards
        return self.reward_history[-num_steps:]


class ObservationBuffer:
    """Handles observation history for environments that use multiple timesteps."""

    def __init__(self, num_obs: int, include_history_steps: int) -> None:
        """Initialize observation buffer.

        Args:
            num_obs: Number of observations per timestep
            include_history_steps: Number of timesteps to include in history
        """
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.num_obs_total = num_obs * include_history_steps
        self.obs_buf = np.zeros(self.num_obs_total)

    def reset(self, new_obs) -> None:
        """Reset buffer with new observation."""
        self.obs_buf = np.tile(new_obs, self.include_history_steps)

    def insert(self, new_obs):
        """Insert new observation and shift history."""
        self.obs_buf[: -self.num_obs] = self.obs_buf[self.num_obs :]
        self.obs_buf[-self.num_obs :] = new_obs

    def get_obs_vec(self, obs_ids=None):
        """Get observation vector for specified timesteps.

        Args:
            obs_ids: Indices of timesteps to include (0 is latest)

        Returns:
            numpy.ndarray: Concatenated observations
        """
        if obs_ids is None:
            obs_ids = np.arange(self.include_history_steps)

        obs = []
        for obs_id in sorted(obs_ids, reverse=True):
            slice_idx = self.include_history_steps - obs_id - 1
            obs.append(
                self.obs_buf[slice_idx * self.num_obs : (slice_idx + 1) * self.num_obs]
            )
        return np.concatenate(obs)


class State:
    """Manages environment state and generates configurable observations."""

    # Define available observation components
    OBSERVATION_COMPONENTS = {
        "projected_gravity": lambda s: s.derived.projected_gravity,
        "projected_gravities": lambda s: s.derived.projected_gravities,
        "ang_vel_body": lambda s: s.raw.ang_vel_body,
        "dof_pos": lambda s: s.raw.dof_pos,
        "dof_vel": lambda s: s.raw.dof_vel,
        # Masked joint observations (for robots with wheels or partial encoders)
        "masked_dof_pos": lambda s: s.masked_dof_pos,
        "masked_dof_vel": lambda s: s.masked_dof_vel,
        "gyros": lambda s: s.raw.gyros,
        "last_action": lambda s: s.action_history.last_action,
        "last_last_action": lambda s: s.action_history.last_last_action,
        "last_last_last_action": lambda s: s.action_history.last_last_last_action,
        "action_history": lambda s: s.action_history.get_history_vector(),
        "last_reward": lambda s: s.reward_history.last_reward,
        "last_last_reward": lambda s: s.reward_history.last_last_reward,
        "last_last_last_reward": lambda s: s.reward_history.last_last_last_reward,
        "reward_history": lambda s: s.reward_history.get_history_vector(),
        "commands": lambda s: s.commands,
        "vel_body": lambda s: s.raw.vel_body,
        "height": lambda s: s.derived.height,
        "heading": lambda s: s.derived.heading,
        "speed": lambda s: s.derived.speed,
        # Simulation-specific components (will return None if not available)
        "mj_data": lambda s: getattr(s, "mj_data", None),
        "mj_model": lambda s: getattr(s, "mj_model", None),
        "contact_geoms": lambda s: getattr(s, "contact_geoms", []),
        "contact_floor_geoms": lambda s: getattr(s, "contact_floor_geoms", []),
        "contact_floor_socks": lambda s: getattr(s, "contact_floor_socks", []),
        "contact_floor_balls": lambda s: getattr(s, "contact_floor_balls", []),
        "num_jointfloor_contact": lambda s: getattr(s, "num_jointfloor_contact", 0),
        "com_vel_world": lambda s: getattr(s, "com_vel_world", np.zeros(3)),
        # Goal distance (from distance sensor)
        "goal_distance": lambda s: np.array([s.raw.goal_distance]),
        "goal_distances": lambda s: s.raw.goal_distances,
    }

    # Define available modular observation components (for per-module observations)
    # These components support indexed access for modular robot architectures
    MODULAR_OBSERVATION_COMPONENTS = {
        # List-type components (use index_type: "element")
        "projected_gravities": lambda s: s.derived.projected_gravities,
        "gyros": lambda s: s.raw.gyros,
        "accs": lambda s: s.raw.accs,
        "quats": lambda s: s.raw.quats,
        # Array-type components (use index_type: "slice")
        "dof_pos": lambda s: s.raw.dof_pos,
        "dof_vel": lambda s: s.raw.dof_vel,
        # Masked joint observations (for robots with wheels or partial encoders)
        "masked_dof_pos": lambda s: s.masked_dof_pos,
        "masked_dof_vel": lambda s: s.masked_dof_vel,
        "last_action": lambda s: s.action_history.last_action,
        "last_last_action": lambda s: s.action_history.last_last_action,
        # Goal distance (from distance sensor) - per module
        "goal_distances": lambda s: s.raw.goal_distances,
    }

    # Define common transformations
    TRANSFORMATIONS = {
        "cos": np.cos,
        "sin": np.sin,
        "normalize": lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x,
        "clip": lambda x: np.clip(x, -1, 1),
        "slice": lambda x, start=None, end=None: x[slice(start, end)],
        "expand_dims": lambda x, axis=0: np.expand_dims(x, axis=axis),
        "flatten": lambda x: x.flatten(),
        "reshape": lambda x, shape: x.reshape(shape),
    }

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize state manager with configurable observation components.

        Args:
            cfg: Configuration object that includes:
                - observation_components: List of components to include
                - observation_transforms: Dict of component transforms
                - observation.modular: Optional modular observation config
        """
        self.cfg = cfg

        self.num_act = cfg.control.num_actions
        self.num_envs = cfg.environment.num_envs
        self.include_history_steps = cfg.observation.include_history_steps
        # self.num_obs = cfg.observation.num_obs
        self.clip_observations = cfg.observation.clip_observations
        self.gravity_vec = np.array(cfg.observation.gravity_vec)
        self.forward_vec = np.array(cfg.observation.forward_vec)
        self.projected_forward_vec = None
        self.dt = cfg.control.dt

        # State containers
        self.raw = RawState(num_dof=self.num_act)
        self.derived = DerivedState(num_dof=self.num_act)
        self.accurate = AccurateState()  # Only used in simulation environments

        # Setup observation masking for joints (e.g., wheels without encoders)
        self._setup_obs_masking(cfg)

        # Action history
        action_history_steps = getattr(
            cfg.observation, "action_history_steps", 3
        )  # Default to 3 steps
        self.action_history = ActionHistoryBuffer(self.num_act, action_history_steps)

        # Reward history
        reward_history_steps = getattr(
            cfg.observation, "reward_history_steps", 3
        )  # Default to 3 steps
        self.reward_history = RewardHistoryBuffer(reward_history_steps)

        # Reward history
        reward_history_steps = getattr(
            cfg.observation, "reward_history_steps", 3
        )  # Default to 3 steps
        self.reward_history = RewardHistoryBuffer(reward_history_steps)

        # Commands

        # Tracking and visualization
        self.observable_data = {}
        self.step_counter = 0
        self.render_lookat_filter = AverageFilter(10)

        # Initialize reward-related state
        self.reset_reward_state()

        # Modular observation settings
        self.modular_mode = False
        self.num_modules = 0
        self.modular_components = []  # Per-module components
        self.global_components = []   # Global components (added once)

        # Set up observation components (flat or modular)
        self.observation_components = []
        self._setup_observation_components()

        # Calculate total observation size
        obs = self._construct_observation()
        self.num_obs = len(obs)

        # Initialize observation buffer
        self.obs_buf = ObservationBuffer(
            self.num_obs * self.num_envs, self.include_history_steps
        )
        
        # Critic observation settings (for asymmetric actor-critic)
        self.critic_observation_components = []
        self.critic_modular_mode = False
        self.critic_modular_components = []
        self.critic_global_components = []
        self.critic_main_module_index = None
        self._setup_critic_observation_components()
        
        # Calculate critic observation size
        critic_obs = self._construct_critic_observation()
        self.num_critic_obs = len(critic_obs)
        
        # Initialize critic observation buffer
        self.critic_obs_buf = ObservationBuffer(
            self.num_critic_obs * self.num_envs, self.include_history_steps
        )

    def __getattr__(self, name):
        """Automatically forward attribute access to raw, derived, accurate state objects, or observable_data.

        This allows accessing state.vel_body instead of state.raw.vel_body.
        Priority: raw state -> derived state -> accurate state -> observable_data -> AttributeError
        """
        # Try raw state first
        if hasattr(self.raw, name):
            return getattr(self.raw, name)

        # Then try derived state
        if hasattr(self.derived, name):
            return getattr(self.derived, name)

        # Then try accurate state (with accurate_ prefix handling)
        if hasattr(self.accurate, name):
            return getattr(self.accurate, name)

        # Handle accurate_ prefixed attributes
        if name.startswith("accurate_"):
            attr_name = name.replace("accurate_", "")
            if hasattr(self.accurate, attr_name):
                return getattr(self.accurate, attr_name)

        # Try action_history attributes
        if hasattr(self, "action_history") and hasattr(self.action_history, name):
            return getattr(self.action_history, name)

        # Try reward_history attributes
        if hasattr(self, "reward_history") and hasattr(self.reward_history, name):
            return getattr(self.reward_history, name)

        # Try observable_data (for mj_data, mj_model, etc.)
        if hasattr(self, "observable_data") and name in self.observable_data:
            return self.observable_data[name]

        # For simulation data that might not be available, return None instead of raising error
        simulation_keys = [
            "mj_data",
            "mj_model",
            "contact_geoms",
            "contact_floor_geoms",
            "contact_floor_socks",
            "contact_floor_balls",
            "num_jointfloor_contact",
            "com_vel_world",
            "adjusted_forward_vec",
        ]
        if name in simulation_keys:
            return None

        # If not found in any state container, raise AttributeError
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value) -> None:
        """Handle attribute setting with forwarding to state objects when appropriate."""
        # Always allow setting of State's own attributes during initialization
        if (
            name
            in [
                "cfg",
                "num_act",
                "num_envs",
                "include_history_steps",
                "clip_observations",
                "gravity_vec",
                "forward_vec",
                "projected_forward_vec",
                "dt",
                "raw",
                "derived",
                "accurate",
                "commands",
                "observable_data",
                "step_counter",
                "action_history",
                "reward_history",
                "render_lookat_filter",
                "observation_components",
                "num_obs",
                "obs_buf",
                "ang_vel_history",
                "vel_history",
                "pos_history",
                "last_dof_vel",
                "contact_counter",
                "fly_counter",
                "jump_timer",
                "vel_filter",
                # Modular observation attributes
                "modular_mode",
                "num_modules",
                "modular_components",
                "global_components",
                # Critic observation attributes
                "critic_observation_components",
                "critic_modular_mode",
                "critic_modular_components",
                "critic_global_components",
                "critic_main_module_index",
                "num_critic_obs",
                "critic_obs_buf",
                # Observation masking attributes
                "dof_pos_mask",
                "dof_vel_mask",
            ]
            or not hasattr(self, "raw")
            or not hasattr(self, "derived")
            or not hasattr(self, "accurate")
        ):
            super().__setattr__(name, value)
            return

        # Handle accurate_ prefixed attributes
        if name.startswith("accurate_"):
            attr_name = name.replace("accurate_", "")
            if hasattr(self.accurate, attr_name):
                setattr(self.accurate, attr_name, value)
                return

        # If the attribute exists in raw state, set it there
        if hasattr(self.raw, name):
            setattr(self.raw, name, value)
        # If the attribute exists in derived state, set it there
        elif hasattr(self.derived, name):
            setattr(self.derived, name, value)
        # If the attribute exists in accurate state, set it there
        elif hasattr(self.accurate, name):
            setattr(self.accurate, name, value)
        # Otherwise, set it on State itself
        else:
            super().__setattr__(name, value)

    def _setup_obs_masking(self, cfg: OmegaConf) -> None:
        """Setup observation masking for joints without full encoder data.
        
        This is useful for robots with wheel joints where position encoder data
        may be unavailable or meaningless (continuous rotation). Allows excluding
        specific joints from dof_pos and/or dof_vel observations.
        
        Configuration example:
            control:
              obs_masking:
                enabled: true
                exclude_joints:
                  dof_pos: [2, 4]  # Exclude joints 2 and 4 from dof_pos
                  dof_vel: []      # Include all joints in dof_vel
        """
        obs_masking = getattr(cfg.control, 'obs_masking', None)
        
        if obs_masking and obs_masking.get('enabled', False):
            exclude_cfg = obs_masking.get('exclude_joints', {})
            self.dof_pos_mask = self._create_joint_mask(
                exclude_cfg.get('dof_pos', []), self.num_act
            )
            self.dof_vel_mask = self._create_joint_mask(
                exclude_cfg.get('dof_vel', []), self.num_act
            )
        else:
            self.dof_pos_mask = None
            self.dof_vel_mask = None
    
    def _create_joint_mask(
        self, exclude_indices: list, num_joints: int
    ) -> Optional[np.ndarray]:
        """Create a boolean mask for joint observations.
        
        Args:
            exclude_indices: List of joint indices to exclude
            num_joints: Total number of joints
            
        Returns:
            Boolean mask array where True = include, False = exclude.
            Returns None if no joints are excluded.
        """
        if not exclude_indices:
            return None
        
        # Validate indices
        for idx in exclude_indices:
            if idx >= num_joints:
                raise ValueError(
                    f"Exclude joint index {idx} >= num_joints {num_joints}"
                )
        
        mask = np.ones(num_joints, dtype=bool)
        mask[exclude_indices] = False
        return mask
    
    @property
    def masked_dof_pos(self) -> np.ndarray:
        """Get dof_pos with excluded joints removed.
        
        Returns:
            Joint positions with masked joints excluded. If no masking is
            configured, returns the full dof_pos array.
        """
        if self.dof_pos_mask is None:
            return self.raw.dof_pos
        return self.raw.dof_pos[self.dof_pos_mask]
    
    @property
    def masked_dof_vel(self) -> np.ndarray:
        """Get dof_vel with excluded joints removed.
        
        Returns:
            Joint velocities with masked joints excluded. If no masking is
            configured, returns the full dof_vel array.
        """
        if self.dof_vel_mask is None:
            return self.raw.dof_vel
        return self.raw.dof_vel[self.dof_vel_mask]

    def _setup_observation_components(self) -> None:
        """Set up observation components based on config.
        
        Supports two modes:
        1. Flat mode (default): Standard observation components concatenated
        2. Modular mode: Per-module observations repeated for each module
        
        Modular mode is enabled by setting observation.modular.enabled = true
        """
        # Check if modular mode is enabled
        modular_cfg = getattr(self.cfg.observation, "modular", None)
        if modular_cfg is not None and getattr(modular_cfg, "enabled", False):
            self._setup_modular_observation_components(modular_cfg)
        else:
            self._setup_flat_observation_components()

    def _setup_flat_observation_components(self) -> None:
        """Set up standard flat observation components."""
        # Get observation components from config
        if not hasattr(self.cfg.observation, "components"):
            # Default observation components if not specified
            components = [
                {"name": "projected_gravity"},
                {"name": "ang_vel_body"},
                {"name": "dof_pos", "transform": "cos"},
                {"name": "dof_vel"},
                {"name": "last_action"},
            ]
        else:
            components = self.cfg.observation.components

        # Process each component
        for comp_spec in components:
            if isinstance(comp_spec, (str, list, tuple)):
                # Handle legacy format (name, transform)
                if isinstance(comp_spec, str):
                    comp_name, transform_name = comp_spec, None
                else:
                    comp_name, transform_name = comp_spec
                comp_spec = {"name": comp_name, "transform": transform_name}

            # Get component name and data function
            comp_name = comp_spec["name"]
            if comp_name not in self.OBSERVATION_COMPONENTS:
                raise ValueError(f"Unknown observation component: {comp_name}")
            data_fn = self.OBSERVATION_COMPONENTS[comp_name]

            # Build transform function
            transform_fn = self._build_transform_fn(comp_spec)

            # Create and add component
            self.observation_components.append(
                ObservationComponent(comp_name, data_fn, transform_fn)
            )

    def _setup_modular_observation_components(self, modular_cfg) -> None:
        """Set up modular observation components for per-module observations.
        
        This enables observation patterns like:
            obs = []
            for i in range(N_MODULES):
                module_obs = np.concatenate([
                    s.projected_gravities[i],
                    s.gyros[i],
                    s.dof_pos[i:i+1],
                    s.dof_vel[i:i+1]
                ])
                obs.append(module_obs)
            obs = np.concatenate(obs)
        
        When main_module_index is set, it creates a single-module observation:
            obs = np.concatenate([
                s.projected_gravities[i],  # Only main module
                s.gyros[i],                # Only main module
                s.dof_pos,                 # ALL (global)
                s.dof_vel,                 # ALL (global)
                s.last_action              # ALL (global)
            ])
        
        Args:
            modular_cfg: Configuration for modular observations with:
                - num_modules: Number of modules (or "auto" to detect)
                - main_module_index: Optional index of main module. When set,
                    per_module_components only use this single module index,
                    creating a "main module + global" observation pattern.
                - per_module_components: List of per-module component specs
                - global_components: Optional list of global component specs
        """
        self.modular_mode = True
        
        # Check for main_module_index mode (single module + global pattern)
        self.main_module_index = getattr(modular_cfg, "main_module_index", None)
        
        # Get number of modules
        num_modules = getattr(modular_cfg, "num_modules", "auto")
        if num_modules == "auto":
            # Auto-detect from num_actions (assuming 1 action per module)
            self.num_modules = self.num_act
        else:
            self.num_modules = int(num_modules)
        
        # Set up per-module components
        per_module_specs = getattr(modular_cfg, "per_module_components", [])
        for comp_spec in per_module_specs:
            comp_spec = self._normalize_component_spec(comp_spec)
            comp_name = comp_spec["name"]
            
            # Get data function from modular components registry
            if comp_name not in self.MODULAR_OBSERVATION_COMPONENTS:
                raise ValueError(
                    f"Unknown modular observation component: {comp_name}. "
                    f"Available: {list(self.MODULAR_OBSERVATION_COMPONENTS.keys())}"
                )
            data_fn = self.MODULAR_OBSERVATION_COMPONENTS[comp_name]
            
            # Get indexing type and slice size
            index_type = comp_spec.get("index_type", "element")
            slice_size = comp_spec.get("slice_size", 1)
            
            # Build transform function
            transform_fn = self._build_transform_fn(comp_spec)
            # Create modular component
            self.modular_components.append(
                ModularObservationComponent(
                    comp_name, data_fn, index_type, slice_size, transform_fn
                )
            )
        
        # Set up global components (added once, not per-module)
        global_specs = getattr(modular_cfg, "global_components", [])
        for comp_spec in global_specs:
            comp_spec = self._normalize_component_spec(comp_spec)
            comp_name = comp_spec["name"]
            
            if comp_name not in self.OBSERVATION_COMPONENTS:
                raise ValueError(f"Unknown observation component: {comp_name}")
            data_fn = self.OBSERVATION_COMPONENTS[comp_name]
            
            transform_fn = self._build_transform_fn(comp_spec)
            
            self.global_components.append(
                ObservationComponent(comp_name, data_fn, transform_fn)
            )

    def _normalize_component_spec(self, comp_spec) -> dict:
        """Normalize component specification to dictionary format."""
        if isinstance(comp_spec, str):
            return {"name": comp_spec}
        elif isinstance(comp_spec, (list, tuple)):
            if len(comp_spec) >= 2:
                return {"name": comp_spec[0], "transform": comp_spec[1]}
            return {"name": comp_spec[0]}
        return dict(comp_spec) if hasattr(comp_spec, "items") else comp_spec

    def _build_transform_fn(self, comp_spec) -> Callable:
        """Build transform function from component specification."""
        def identity(x):
            return x
        
        if "transform" not in comp_spec or comp_spec.get("transform") is None:
            return identity
        
        transform_spec = comp_spec["transform"]
        
        if isinstance(transform_spec, str):
            # Single transform
            if transform_spec not in self.TRANSFORMATIONS:
                raise ValueError(f"Unknown transform: {transform_spec}")
            return self.TRANSFORMATIONS[transform_spec]
        
        elif isinstance(transform_spec, list):
            # Chain of transforms
            transforms = []
            for t in transform_spec:
                if isinstance(t, str):
                    if t not in self.TRANSFORMATIONS:
                        raise ValueError(f"Unknown transform: {t}")
                    transforms.append(self.TRANSFORMATIONS[t])
                elif isinstance(t, dict):
                    # Transform with parameters
                    t_name = t["name"]
                    if t_name not in self.TRANSFORMATIONS:
                        raise ValueError(f"Unknown transform: {t_name}")
                    t_params = {k: v for k, v in t.items() if k != "name"}
                    transforms.append(
                        lambda x, fn=self.TRANSFORMATIONS[t_name], params=t_params: fn(x, **params)
                    )
            
            def chain_transform(x, transforms=transforms):
                for t in transforms:
                    x = t(x)
                return x
            
            return chain_transform
        
        return identity

    def _setup_critic_observation_components(self) -> None:
        """Set up critic observation components for asymmetric actor-critic.
        
        Critic observations can include privileged information that the policy
        doesn't have access to (e.g., true velocities, contact info).
        
        If no critic config is specified, critic observations default to
        the same as policy observations.
        
        Configuration example in YAML:
            observation:
              critic:
                enabled: true
                # Option 1: Flat mode - list of components
                components:
                  - name: projected_gravity
                  - name: ang_vel_body
                  - name: vel_body        # Privileged info!
                  - name: dof_pos
                  - name: dof_vel
                  - name: last_action
                
                # Option 2: Modular mode (same as policy modular config)
                # modular:
                #   enabled: true
                #   num_modules: 3
                #   per_module_components:
                #     - name: projected_gravities
                #       index_type: element
                #   global_components:
                #     - name: vel_body      # Privileged info!
                #     - name: dof_pos
        """
        critic_cfg = getattr(self.cfg.observation, "critic", None)
        
        # If no critic config, default to copying policy observations
        if critic_cfg is None or not getattr(critic_cfg, "enabled", False):
            # Copy policy observation setup
            self.critic_observation_components = list(self.observation_components)
            self.critic_modular_mode = self.modular_mode
            self.critic_modular_components = list(self.modular_components)
            self.critic_global_components = list(self.global_components)
            self.critic_main_module_index = getattr(self, 'main_module_index', None)
            return
        
        # Check if modular mode is enabled for critic
        critic_modular_cfg = getattr(critic_cfg, "modular", None)
        if critic_modular_cfg is not None and getattr(critic_modular_cfg, "enabled", False):
            self._setup_critic_modular_components(critic_modular_cfg)
        else:
            self._setup_critic_flat_components(critic_cfg)
    
    def _setup_critic_flat_components(self, critic_cfg) -> None:
        """Set up flat critic observation components."""
        components = getattr(critic_cfg, "components", None)
        
        if components is None:
            # If no components specified, copy from policy
            self.critic_observation_components = list(self.observation_components)
            return
        
        self.critic_modular_mode = False
        for comp_spec in components:
            comp_spec = self._normalize_component_spec(comp_spec)
            comp_name = comp_spec["name"]
            
            if comp_name not in self.OBSERVATION_COMPONENTS:
                raise ValueError(f"Unknown critic observation component: {comp_name}")
            data_fn = self.OBSERVATION_COMPONENTS[comp_name]
            transform_fn = self._build_transform_fn(comp_spec)
            
            self.critic_observation_components.append(
                ObservationComponent(comp_name, data_fn, transform_fn)
            )
    
    def _setup_critic_modular_components(self, critic_modular_cfg) -> None:
        """Set up modular critic observation components."""
        self.critic_modular_mode = True
        
        # Main module index for critic
        self.critic_main_module_index = getattr(critic_modular_cfg, "main_module_index", None)
        
        # Number of modules (use same as policy if not specified)
        num_modules = getattr(critic_modular_cfg, "num_modules", "auto")
        if num_modules == "auto":
            # Use the same num_modules as policy
            num_modules = self.num_modules if self.num_modules > 0 else self.num_act
        
        # Per-module components for critic
        per_module_specs = getattr(critic_modular_cfg, "per_module_components", [])
        for comp_spec in per_module_specs:
            comp_spec = self._normalize_component_spec(comp_spec)
            comp_name = comp_spec["name"]
            
            if comp_name not in self.MODULAR_OBSERVATION_COMPONENTS:
                raise ValueError(
                    f"Unknown critic modular observation component: {comp_name}. "
                    f"Available: {list(self.MODULAR_OBSERVATION_COMPONENTS.keys())}"
                )
            data_fn = self.MODULAR_OBSERVATION_COMPONENTS[comp_name]
            index_type = comp_spec.get("index_type", "element")
            slice_size = comp_spec.get("slice_size", 1)
            transform_fn = self._build_transform_fn(comp_spec)
            
            self.critic_modular_components.append(
                ModularObservationComponent(
                    comp_name, data_fn, index_type, slice_size, transform_fn
                )
            )
        
        # Global components for critic
        global_specs = getattr(critic_modular_cfg, "global_components", [])
        for comp_spec in global_specs:
            comp_spec = self._normalize_component_spec(comp_spec)
            comp_name = comp_spec["name"]
            
            if comp_name not in self.OBSERVATION_COMPONENTS:
                raise ValueError(f"Unknown critic observation component: {comp_name}")
            data_fn = self.OBSERVATION_COMPONENTS[comp_name]
            transform_fn = self._build_transform_fn(comp_spec)
            
            self.critic_global_components.append(
                ObservationComponent(comp_name, data_fn, transform_fn)
            )

    def reset(self) -> None:
        """Reset state variables."""
        # # Reset raw state
        # self.raw = RawState(num_dof=self.num_act)

        # # Reset derived state
        # self.derived = DerivedState()

        # # Reset accurate state
        # self.accurate = AccurateState()

        # Reset action history
        action_history_steps = getattr(self.cfg.observation, "action_history_steps", 3)
        self.action_history = ActionHistoryBuffer(self.num_act, action_history_steps)

        # Reset reward history
        reward_history_steps = getattr(self.cfg.observation, "reward_history_steps", 3)
        self.reward_history = RewardHistoryBuffer(reward_history_steps)

        # Reset commands
        # self.commands = np.zeros(3)

        # Reset tracking
        # self.observable_data = {}
        self.step_counter = 0
        self.render_lookat_filter = AverageFilter(10)

        # Reset reward-related state
        self.reset_reward_state()

    def reset_reward_state(self) -> None:
        """Reset state variables used for reward calculation."""
        self.ang_vel_history = []
        self.vel_history = []
        self.pos_history = []
        self.last_dof_vel = np.zeros(self.num_act)
        self.contact_counter = {}
        self.fly_counter = 0
        self.jump_timer = 0
        self.vel_filter = AverageFilter(int(0.5 / self.dt))

    def set_command_manager(self, command_manager) -> None:
        """Set the command manager reference for named command access.

        Args:
            command_manager: The CommandManager instance from the environment
        """
        self.command_manager = command_manager

    def get_command_by_name(self, name: str):
        """Get a command value by name.

        Args:
            name: Command dimension name

        Returns:
            Current value of the named command

        Raises:
            AttributeError: If no command manager is set
            ValueError: If command name is not found
        """
        if not hasattr(self, "command_manager") or self.command_manager is None:
            raise AttributeError(
                "No command manager set. Call set_command_manager() first."
            )
        return self.command_manager.get_command_by_name(name)

    def set_command_by_name(self, name: str, value: float) -> None:
        """Set a command value by name.

        Args:
            name: Command dimension name
            value: New command value

        Raises:
            AttributeError: If no command manager is set
            ValueError: If command name is not found
        """
        if not hasattr(self, "command_manager") or self.command_manager is None:
            raise AttributeError(
                "No command manager set. Call set_command_manager() first."
            )
        self.command_manager.set_command_by_name(name, value)

    def get_commands_dict(self):
        """Get all commands as a dictionary mapping names to values.

        Returns:
            Dictionary with command names as keys and current values as values

        Raises:
            AttributeError: If no command manager is set
        """
        if not hasattr(self, "command_manager") or self.command_manager is None:
            raise AttributeError(
                "No command manager set. Call set_command_manager() first."
            )
        return self.command_manager.get_commands_dict()

    @property
    def command_names(self):
        """Get list of available command names.

        Returns:
            List of command dimension names

        Raises:
            AttributeError: If no command manager is set
        """
        if not hasattr(self, "command_manager") or self.command_manager is None:
            raise AttributeError(
                "No command manager set. Call set_command_manager() first."
            )
        return self.command_manager.command_names

    @property
    def commands(self):
        """Get current command values.

        Returns:
            numpy.ndarray: Current command values

        Raises:
            AttributeError: If no command manager is set
        """
        if not hasattr(self, "command_manager") or self.command_manager is None:
            num_commands = len(
                getattr(self.cfg.task.commands, "dimensions", {})
            )  # Default to 0 if not specified
            return np.zeros(num_commands)
        return np.array(list(self.command_manager.get_commands_dict().values()))

    def update(self, data: dict[str, Any]) -> None:
        """Update state with new data.

        Args:
            data: Dictionary of new state data
        """
        # Update raw state
        self.raw.update(data)

        # Update accurate state (simulation only)
        self.accurate.update(data)

        # Update action history if last_action is provided
        if "last_action" in data:
            self.action_history.update(data["last_action"])

        # Update reward history if last_reward is provided
        if "last_reward" in data:
            self.reward_history.update(data["last_reward"])

        # Update reward history if reward is provided
        if "reward" in data:
            self.reward_history.update(data["reward"])

        # Update position history for speed calculation
        self.pos_history.append(self.raw.pos_world.copy())
        # Keep only a limited history (e.g., last 10 steps for speed calculation)
        max_history_length = 1000
        if len(self.pos_history) > max_history_length:
            self.pos_history.pop(0)

        # Compute derived state
        self._compute_derived_state()

        # Update observable data
        self.observable_data = data.copy()
        self.observable_data.update(
            {
                "projected_gravity": self.derived.projected_gravity,
                "heading": self.derived.heading,
                "dof_pos": self.raw.dof_pos,
                "dof_vel": self.raw.dof_vel,
            }
        )

        # Update step counter
        self.step_counter += 1

    def _compute_derived_state(self) -> None:
        """Compute derived state from raw state."""
        # Height
        self.derived.height = np.expand_dims(self.raw.pos_world[2], axis=0)

        # Projected gravity
        self.derived.projected_gravity = quat_rotate_inverse(
            self.raw.quat, self.gravity_vec
        )

        # Projected gravities for each quaternion
        self.derived.projected_gravities = [
            quat_rotate_inverse(quat, self.gravity_vec) for quat in self.raw.quats
        ]
        if not self.derived.projected_gravities:
            print("Warning: No projected gravities available")

        # Heading
        forward = quat_apply(self.raw.quat, self.forward_vec)
        self.derived.heading = np.expand_dims(
            np.arctan2(forward[1], forward[0]), axis=0
        )

        # Speed (using position history)
        if len(self.pos_history) >= 2:
            # Calculate speed using current position and N steps back
            # Use configuration or default to 1 step back
            speed_calculation_steps = getattr(
                self.cfg.observation, "speed_calculation_steps", 100
            )
            steps_back = min(speed_calculation_steps, len(self.pos_history) - 1)

            current_pos = self.pos_history[-1]
            past_pos = self.pos_history[-(steps_back + 1)]

            # Calculate displacement and speed
            displacement = current_pos - past_pos
            distance = np.linalg.norm(displacement)
            # Speed = distance / (time_steps * dt)
            time_elapsed = steps_back * self.dt
            speed = distance / time_elapsed if time_elapsed > 0 else 0.0

            self.derived.speed = np.expand_dims(speed, axis=0)
        else:
            # Not enough history yet, set speed to 0
            self.derived.speed = np.zeros(1)

    def get_observation(self, insert=True, reset=False):
        """Get observation vector based on current state.

        Args:
            insert: Whether to insert observation into buffer
            reset: Whether to reset observation buffer

        Returns:
            numpy.ndarray: Observation vector
        """
        obs = self._construct_observation()
        obs = np.clip(obs, -self.clip_observations, self.clip_observations)

        if reset:
            self.obs_buf.reset(obs)
        elif insert:
            # print(f"!!Inserting observation of size {obs.shape}")
            self.obs_buf.insert(obs)

        return self.obs_buf.get_obs_vec()

    def get_critic_observation(self, insert=True, reset=False):
        """Get critic observation vector based on current state.
        
        Critic observations can include privileged information not available
        to the policy (e.g., true velocities, contact info).

        Args:
            insert: Whether to insert observation into buffer
            reset: Whether to reset observation buffer

        Returns:
            numpy.ndarray: Critic observation vector
        """
        critic_obs = self._construct_critic_observation()
        critic_obs = np.clip(critic_obs, -self.clip_observations, self.clip_observations)

        if reset:
            self.critic_obs_buf.reset(critic_obs)
        elif insert:
            self.critic_obs_buf.insert(critic_obs)

        return self.critic_obs_buf.get_obs_vec()
    
    def _construct_critic_observation(self):
        """Construct critic observation vector based on critic components.

        Returns:
            numpy.ndarray: Raw critic observation vector
        """
        if self.critic_modular_mode:
            return self._construct_critic_modular_observation()
        else:
            return self._construct_critic_flat_observation()
    
    def _construct_critic_flat_observation(self):
        """Construct flat critic observation vector.
        
        Returns:
            numpy.ndarray: Raw critic observation vector
        """
        obs_parts = []
        for component in self.critic_observation_components:
            data = component.get_data(self)
            flattened_data = np.asarray(data).flatten()
            obs_parts.append(flattened_data)
        
        if not obs_parts:
            return np.array([])
        
        return np.concatenate(obs_parts)
    
    def _construct_critic_modular_observation(self):
        """Construct modular critic observation vector.
        
        Returns:
            numpy.ndarray: Raw critic observation vector with modular structure
        """
        obs_parts = []
        
        # Determine which module indices to iterate over
        if self.critic_main_module_index is not None:
            module_indices = [self.critic_main_module_index]
        else:
            module_indices = range(self.num_modules)
        
        # Per-module observations for critic
        for module_idx in module_indices:
            module_obs_parts = []
            for component in self.critic_modular_components:
                try:
                    data = component.get_data_for_module(self, module_idx)
                    flattened_data = np.asarray(data).flatten()
                    module_obs_parts.append(flattened_data)
                except (IndexError, TypeError) as e:
                    print(
                        f"ERROR: Failed to get critic data for component '{component.name}' "
                        f"at module index {module_idx}: {e}"
                    )
                    raise
            
            if module_obs_parts:
                obs_parts.append(np.concatenate(module_obs_parts))
        
        # Global observations for critic (added once at the end)
        for component in self.critic_global_components:
            data = component.get_data(self)
            flattened_data = np.asarray(data).flatten()
            obs_parts.append(flattened_data)
        
        if not obs_parts:
            return np.array([])
        
        return np.concatenate(obs_parts)

    def _construct_observation(self):
        """Construct observation vector based on components.

        Returns:
            numpy.ndarray: Raw observation vector
            
        Supports two modes:
        1. Flat mode: Standard concatenation of all observation components
        2. Modular mode: Per-module observations repeated for each module,
           followed by global components
        """
        if self.modular_mode:
            return self._construct_modular_observation()
        else:
            return self._construct_flat_observation()

    def _construct_flat_observation(self):
        """Construct flat observation vector (standard mode).
        
        Returns:
            numpy.ndarray: Raw observation vector
        """
        obs_parts = []
        for component in self.observation_components:
            data = component.get_data(self)
            flattened_data = np.asarray(data).flatten()

            # Check for NaN values in this component
            if np.any(np.isnan(flattened_data)):
                nan_indices = np.where(np.isnan(flattened_data))[0]
                print(
                    f"WARNING: NaN detected in observation component '{component.name}'"
                )
                print(f"  Original data shape: {np.asarray(data).shape}")
                print(f"  Flattened data shape: {flattened_data.shape}")
                print(f"  NaN indices: {nan_indices}")
                print(f"  Data sample: {flattened_data[:min(10, len(flattened_data))]}")

            obs_parts.append(flattened_data)

        # Ensure 1D array for concatenation
        observation = np.concatenate(obs_parts)

        # Final check for any remaining NaN values in the complete observation
        if np.any(np.isnan(observation)):
            nan_count = np.sum(np.isnan(observation))
            total_count = len(observation)
            print(
                f"ERROR: Final observation contains {nan_count}/{total_count} NaN values!"
            )
            print(f"Step counter: {self.step_counter}")

        return observation

    def _construct_modular_observation(self):
        """Construct modular observation vector (per-module mode).
        
        Generates observations in the pattern:
            obs = []
            for i in range(N_MODULES):
                module_obs = np.concatenate([
                    component.get_data_for_module(state, i)
                    for component in per_module_components
                ])
                obs.append(module_obs)
            obs.append(global_obs)  # Global components added once
            obs = np.concatenate(obs)
        
        When main_module_index is set, generates:
            obs = np.concatenate([
                per_module_data_for_main_module,  # Only main module
                global_obs                         # All global components
            ])
        
        Returns:
            numpy.ndarray: Raw observation vector with modular structure
        """
        obs_parts = []
        
        # Determine which module indices to iterate over
        if self.main_module_index is not None:
            # Main module mode: only use the specified main module index
            module_indices = [self.main_module_index]
        else:
            # Standard mode: iterate over all modules
            module_indices = range(self.num_modules)
        
        # Per-module observations
        for module_idx in module_indices:
            module_obs_parts = []
            for component in self.modular_components:
                # print(f"Component: {component.name}")
                try:
                    data = component.get_data_for_module(self, module_idx)
                    flattened_data = np.asarray(data).flatten()
                    
                    # Check for NaN values
                    if np.any(np.isnan(flattened_data)):
                        print(
                            f"WARNING: NaN in modular component '{component.name}' "
                            f"for module {module_idx}"
                        )
                    
                    module_obs_parts.append(flattened_data)
                except (IndexError, TypeError) as e:
                    print(
                        f"ERROR: Failed to get data for component '{component.name}' "
                        f"at module index {module_idx}: {e}"
                    )
                    raise
            
            if module_obs_parts:
                obs_parts.append(np.concatenate(module_obs_parts))
        
        # Global observations (added once at the end)
        for component in self.global_components:
            data = component.get_data(self)
            flattened_data = np.asarray(data).flatten()
            
            if np.any(np.isnan(flattened_data)):
                print(
                    f"WARNING: NaN in global component '{component.name}'"
                )
            
            obs_parts.append(flattened_data)
        
        # Concatenate all parts
        if not obs_parts:
            return np.array([])
        
        observation = np.concatenate(obs_parts)
        
        # Final NaN check
        if np.any(np.isnan(observation)):
            nan_count = np.sum(np.isnan(observation))
            total_count = len(observation)
            print(
                f"ERROR: Modular observation contains {nan_count}/{total_count} NaN values!"
            )
            print(f"Step counter: {self.step_counter}")
        
        return observation

    def get_modular_observation_info(self) -> dict:
        """Get information about the modular observation structure.
        
        Returns:
            Dictionary with modular observation details including:
            - modular_mode: Whether modular mode is enabled
            - num_modules: Number of modules
            - per_module_obs_size: Size of observation per module
            - global_obs_size: Size of global observations
            - total_obs_size: Total observation size
            - component_names: Names of components in order
        """
        if not self.modular_mode:
            return {
                "modular_mode": False,
                "num_modules": 0,
                "per_module_obs_size": 0,
                "global_obs_size": self.num_obs,
                "total_obs_size": self.num_obs,
                "component_names": [c.name for c in self.observation_components],
            }
        
        # Calculate per-module observation size
        per_module_size = 0
        per_module_names = []
        for comp in self.modular_components:
            # Get sample data for module 0 to determine size
            try:
                sample = comp.get_data_for_module(self, 0)
                per_module_size += np.asarray(sample).flatten().shape[0]
                per_module_names.append(comp.name)
            except Exception:
                pass
        
        # Calculate global observation size
        global_size = 0
        global_names = []
        for comp in self.global_components:
            try:
                sample = comp.get_data(self)
                global_size += np.asarray(sample).flatten().shape[0]
                global_names.append(comp.name)
            except Exception:
                pass
        
        return {
            "modular_mode": True,
            "num_modules": self.num_modules,
            "per_module_obs_size": per_module_size,
            "global_obs_size": global_size,
            "total_obs_size": per_module_size * self.num_modules + global_size,
            "per_module_components": per_module_names,
            "global_components": global_names,
        }

    def get_custom_commands(self, command_type):
        """Get custom commands based on command type."""
        info = {}
        if command_type == "onehot_dirichlet":
            if self.step_counter < self.cfg.trainer.total_steps / 2:
                commands = np.zeros(3)
                commands[np.random.randint(3)] = 1
            else:
                commands = np.random.dirichlet(np.ones(3))
        elif command_type == "onehot":
            commands = np.zeros(3)
            commands[np.random.randint(3)] = 1
        return commands, info

    def get_available_attributes(self):
        """Get a list of all available state attributes for debugging/inspection."""
        state_attrs = []

        # Raw state attributes
        raw_attrs = [attr for attr in dir(self.raw) if not attr.startswith("_")]
        state_attrs.extend([f"raw.{attr}" for attr in raw_attrs])

        # Derived state attributes
        derived_attrs = [attr for attr in dir(self.derived) if not attr.startswith("_")]
        state_attrs.extend([f"derived.{attr}" for attr in derived_attrs])

        # Accurate state attributes
        accurate_attrs = [
            attr
            for attr in dir(self.accurate)
            if not attr.startswith("_")
            and attr not in ["update", "is_available", "reset"]
        ]
        state_attrs.extend([f"accurate.{attr}" for attr in accurate_attrs])

        # Direct State attributes
        direct_attrs = [
            attr
            for attr in dir(self)
            if not attr.startswith("_")
            and attr not in ["raw", "derived", "accurate", "get_available_attributes"]
        ]
        state_attrs.extend([f"direct.{attr}" for attr in direct_attrs])

        return sorted(state_attrs)

    def update_action_history(self, action: np.ndarray) -> None:
        """Manually update action history (useful when actions are applied outside of state updates).

        Args:
            action: New action to add to history
        """
        self.action_history.update(action)

    def get_mujoco_data(self):
        """Get MuJoCo data object if available.

        Returns:
            MuJoCo data object or None if not available
        """
        return getattr(self, "mj_data", None)

    def get_mujoco_model(self):
        """Get MuJoCo model object if available.

        Returns:
            MuJoCo model object or None if not available
        """
        return getattr(self, "mj_model", None)

    def get_simulation_data(self) -> dict[str, Any]:
        """Get all simulation-specific data.

        Returns:
            Dictionary containing simulation data like mj_data, mj_model, etc.
        """
        sim_data = {}
        sim_keys = [
            "mj_data",
            "mj_model",
            "adjusted_forward_vec",
            "contact_geoms",
            "num_jointfloor_contact",
            "contact_floor_geoms",
            "contact_floor_socks",
            "contact_floor_balls",
            "com_vel_world",
        ]

        for key in sim_keys:
            if hasattr(self, key):
                sim_data[key] = getattr(self, key)

        return sim_data

    def get_sensor_data(self) -> dict[str, Any]:
        """Get all sensor data.

        Returns:
            Dictionary containing sensor readings like gyros, accs, etc.
        """
        sensor_data = {}
        sensor_keys = ["gyros", "accs", "quats", "goal_distance", "goal_distances"]

        for key in sensor_keys:
            if hasattr(self, key):
                sensor_data[key] = getattr(self, key)

        return sensor_data

    def get_all_data(self) -> dict[str, Any]:
        """Get all available state data including observable_data.

        Returns:
            Dictionary containing all available data
        """
        all_data = {}

        # Add raw state data
        for attr in [
            "pos_world",
            "quat",
            "quats",
            "vel_body",
            "vel_world",
            "ang_vel_body",
            "ang_vel_world",
            "dof_pos",
            "dof_vel",
            "gyros",
            "accs",
            "goal_distance",
            "goal_distances",
        ]:
            if hasattr(self.raw, attr):
                all_data[f"raw_{attr}"] = getattr(self.raw, attr)

        # Add derived state data
        for attr in [
            "height",
            "projected_gravity",
            "projected_gravities",
            "heading",
            "speed",
        ]:
            if hasattr(self.derived, attr):
                all_data[f"derived_{attr}"] = getattr(self.derived, attr)

        # Add accurate state data
        for attr in ["quat", "vel_body", "vel_world", "pos_world", "ang_vel_body"]:
            if (
                hasattr(self.accurate, attr)
                and getattr(self.accurate, attr) is not None
            ):
                all_data[f"accurate_{attr}"] = getattr(self.accurate, attr)

        # Add observable data
        if hasattr(self, "observable_data"):
            all_data.update(self.observable_data)

        return all_data

    def has_simulation_data(self) -> bool:
        """Check if simulation data (MuJoCo) is available.

        Returns:
            True if simulation data is available
        """
        mj_data = getattr(self, "mj_data", None)
        return mj_data is not None

    def get_data(self, key: str, default=None):
        """Get data with fallback to default.

        Args:
            key: Data key to retrieve
            default: Default value if key not found

        Returns:
            Data value or default
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def get_full_state(self, components: list[str] | None = None) -> dict[str, Any]:
        """Get full/privileged state data for recording or analysis.

        This method provides access to the complete state information that
        may be different from what the policy observes. Useful for recording
        rollouts with privileged information.

        Args:
            components: List of component names to include. If None, returns
                all available state data.

        Returns:
            Dictionary mapping component names to their values.

        Example:
            # Get specific components for recording
            state_data = state.get_full_state([
                "pos_world", "vel_world", "dof_pos", "dof_vel"
            ])

            # Get all available state data
            all_state = state.get_full_state()
        """
        if components is None:
            # Return all available data
            return self.get_all_data()

        result = {}
        for comp_name in components:
            value = self._get_state_component(comp_name)
            if value is not None:
                result[comp_name] = value

        return result

    def _get_state_component(self, comp_name: str) -> Any:
        """Get a specific state component by name.

        Args:
            comp_name: Name of the component to retrieve.

        Returns:
            The component value, or None if not found.
        """
        # Check OBSERVATION_COMPONENTS registry
        if comp_name in self.OBSERVATION_COMPONENTS:
            try:
                return self.OBSERVATION_COMPONENTS[comp_name](self)
            except Exception:
                pass

        # Try direct attribute access
        if hasattr(self, comp_name):
            return getattr(self, comp_name)

        # Try raw state
        if hasattr(self.raw, comp_name):
            return getattr(self.raw, comp_name)

        # Try derived state
        if hasattr(self.derived, comp_name):
            return getattr(self.derived, comp_name)

        # Try accurate state
        if hasattr(self.accurate, comp_name):
            return getattr(self.accurate, comp_name)

        # Try observable_data
        if hasattr(self, 'observable_data') and comp_name in self.observable_data:
            return self.observable_data[comp_name]

        return None

    def get_full_state_vector(self, components: list[str] | None = None) -> np.ndarray:
        """Get full/privileged state as a flattened vector.

        Similar to get_full_state but returns a single numpy array with
        all components concatenated and flattened.

        Args:
            components: List of component names to include. If None, uses
                default privileged components.

        Returns:
            Flattened numpy array of state values.
        """
        if components is None:
            # Default privileged components
            components = [
                "pos_world",
                "quat",
                "vel_world",
                "vel_body",
                "ang_vel_world",
                "ang_vel_body",
                "dof_pos",
                "dof_vel",
                "projected_gravity",
                "height",
                "heading",
                "speed",
            ]

        parts = []
        for comp_name in components:
            value = self._get_state_component(comp_name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    parts.append(value.flatten())
                elif isinstance(value, (list, tuple)):
                    parts.append(np.array(value).flatten())
                elif isinstance(value, (int, float)):
                    parts.append(np.array([value]))

        return np.concatenate(parts) if parts else np.array([])

    @classmethod
    def register_observation_component(cls, name: str, data_fn: Callable):
        """Register a new observation component.

        Args:
            name: Component name for configuration
            data_fn: Function that takes a state object and returns numpy array data

        Example:
            # Register a custom observation component
            def custom_energy(state):
                return np.array([np.sum(np.square(state.dof_vel))])

            State.register_observation_component('energy', custom_energy)

            # Use in configuration:
            observation:
              components:
                - name: energy
                  transform: normalize
        """
        if not callable(data_fn):
            raise ValueError("data_fn must be callable")

        cls.OBSERVATION_COMPONENTS[name] = data_fn

    @classmethod
    def register_modular_observation_component(cls, name: str, data_fn: Callable):
        """Register a new modular observation component.

        Args:
            name: Component name for configuration
            data_fn: Function that takes a state object and returns indexable data
                    (list of arrays for "element" index_type, or array for "slice")

        Example:
            # Register custom per-module sensor data
            def custom_module_sensors(state):
                return state.raw.custom_sensors  # List of arrays per module

            State.register_modular_observation_component('custom_sensors', custom_module_sensors)

            # Use in configuration:
            observation:
              modular:
                enabled: true
                num_modules: 5
                per_module_components:
                  - name: custom_sensors
                    index_type: element
        """
        if not callable(data_fn):
            raise ValueError("data_fn must be callable")

        cls.MODULAR_OBSERVATION_COMPONENTS[name] = data_fn

    @classmethod
    def register_transformation(cls, name: str, transform_fn: Callable):
        """Register a new transformation function.

        Args:
            name: Transform name for configuration
            transform_fn: Function that takes and returns numpy array

        Example:
            # Register a custom transformation
            def square_transform(x):
                return np.square(x)

            State.register_transformation('square', square_transform)

            # Use in configuration:
            observation:
              components:
                - name: dof_pos
                  transform: square
        """
        if not callable(transform_fn):
            raise ValueError("transform_fn must be callable")

        cls.TRANSFORMATIONS[name] = transform_fn

    @classmethod
    def list_observation_components(cls) -> list[str]:
        """Get list of all available observation component names.

        Returns:
            List of component names that can be used in configurations
        """
        return list(cls.OBSERVATION_COMPONENTS.keys())

    @classmethod
    def list_modular_observation_components(cls) -> list[str]:
        """Get list of all available modular observation component names.

        Returns:
            List of component names that can be used in modular configurations
        """
        return list(cls.MODULAR_OBSERVATION_COMPONENTS.keys())

    @classmethod
    def list_transformations(cls) -> list[str]:
        """Get list of all available transformation names.

        Returns:
            List of transformation names that can be used in configurations
        """
        return list(cls.TRANSFORMATIONS.keys())


# Convenience functions for registration
def register_observation_component(name: str, data_fn: Callable):
    """Register a new observation component.

    Args:
        name: Component name for configuration
        data_fn: Function that takes a state object and returns numpy array data

    Example:
        # Register a custom observation component
        def custom_energy(state):
            return np.array([np.sum(np.square(state.dof_vel))])

        register_observation_component('energy', custom_energy)
    """
    State.register_observation_component(name, data_fn)


def register_modular_observation_component(name: str, data_fn: Callable):
    """Register a new modular observation component.

    Args:
        name: Component name for configuration
        data_fn: Function that takes a state object and returns indexable data

    Example:
        # Register custom per-module sensor data
        def custom_module_sensors(state):
            return state.raw.custom_sensors  # List of arrays per module

        register_modular_observation_component('custom_sensors', custom_module_sensors)
    """
    State.register_modular_observation_component(name, data_fn)


def register_transformation(name: str, transform_fn: Callable):
    """Register a new transformation function.

    Args:
        name: Transform name for configuration
        transform_fn: Function that takes and returns numpy array

    Example:
        # Register a custom transformation
        def square_transform(x):
            return np.square(x)

        register_transformation('square', square_transform)
    """
    State.register_transformation(name, transform_fn)


def list_observation_components() -> list[str]:
    """Get list of all available observation component names."""
    return State.list_observation_components()


def list_modular_observation_components() -> list[str]:
    """Get list of all available modular observation component names."""
    return State.list_modular_observation_components()


def list_transformations() -> list[str]:
    """Get list of all available transformation names."""
    return State.list_transformations()
