"""
Rollout Recorder for MetaMachine

This module provides utilities for recording robot rollouts with separate
observations for policy inference and data recording. This is useful when
you want to record privileged/full state information during rollouts while
the policy only sees a subset of observations.

Example:
    >>> from metamachine.utils.rollout_recorder import RolloutRecorder
    >>> 
    >>> # Create recorder with custom recording components
    >>> recorder = RolloutRecorder(
    ...     recording_components=["pos_world", "vel_world", "dof_pos", "dof_vel"],
    ...     include_actions=True,
    ...     include_rewards=True,
    ... )
    >>> 
    >>> # During rollout
    >>> obs, _ = env.reset()
    >>> recorder.start_episode()
    >>> for step in range(num_steps):
    ...     action = policy.predict(obs)
    ...     obs, reward, done, truncated, info = env.step(action)
    ...     recorder.record(env.state, action, reward, info)
    ...     if done or truncated:
    ...         break
    >>> recorder.end_episode()
    >>> 
    >>> # Save data
    >>> recorder.save("rollout_data.npz")

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

from dataclasses import dataclass, field
from pathlib import Path
import pdb
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


@dataclass
class EpisodeData:
    """Container for a single episode's recorded data."""
    
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    custom_data: Dict[str, List[Any]] = field(default_factory=dict)
    component_data: Dict[str, List[Any]] = field(default_factory=dict)  # For separate components
    
    def __len__(self) -> int:
        return len(self.observations) or len(next(iter(self.component_data.values()), []))
    
    def to_dict(self, include_observations: bool = True) -> Dict[str, Any]:
        """Convert episode data to dictionary format.
        
        Args:
            include_observations: Whether to include the concatenated observations array.
                Set to False if using separate_components mode.
        """
        result = {}
        
        if include_observations and self.observations:
            result["observations"] = np.array(self.observations)
        
        if self.actions:
            result["actions"] = np.array(self.actions)
        if self.rewards:
            result["rewards"] = np.array(self.rewards)
        if self.dones:
            result["dones"] = np.array(self.dones)
            
        # Add component data (each component as separate key)
        for key, values in self.component_data.items():
            if values:
                result[key] = np.array(values)
        
        # Add custom data
        for key, values in self.custom_data.items():
            if values:
                result[key] = np.array(values)
                
        return result


class RolloutRecorder:
    """Records robot rollouts with configurable observation components.
    
    This recorder allows you to capture different data during rollouts than
    what the policy uses for inference. For example, you can record full
    state information (positions, velocities, contacts) while the policy
    only sees a processed observation vector.
    
    Attributes:
        recording_components: List of state component names to record
        include_actions: Whether to record actions
        include_rewards: Whether to record rewards
        include_infos: Whether to record info dicts
        custom_extractors: Custom functions to extract additional data
    """
    
    # Default components that provide "privileged" state information
    DEFAULT_RECORDING_COMPONENTS = [
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
        "commands",
    ]
    
    def __init__(
        self,
        recording_components: Optional[List[str]] = None,
        include_actions: bool = True,
        include_rewards: bool = True,
        include_infos: bool = False,
        custom_extractors: Optional[Dict[str, Callable]] = None,
        separate_components: bool = True,
        action_as_actuator_command: bool = False,
    ):
        """Initialize the rollout recorder.
        
        Args:
            recording_components: List of state component names to record.
                If None, uses DEFAULT_RECORDING_COMPONENTS.
                Available components include: pos_world, quat, vel_world,
                vel_body, ang_vel_world, ang_vel_body, dof_pos, dof_vel,
                projected_gravity, height, heading, speed, commands, gyros, etc.
            include_actions: Whether to record actions taken by the policy.
            include_rewards: Whether to record rewards received.
            include_infos: Whether to record full info dictionaries.
            custom_extractors: Dictionary mapping names to functions that
                extract custom data from state. Each function should accept
                a state object and return a value to record.
            separate_components: If True, each recording_component is stored
                as a separate key in the trajectory dict (e.g., 'gyros', 'dof_pos').
                If False (default), all components are concatenated into 'observations'.
            action_as_actuator_command: If True, records action + default_dof_pos
                (the final position command sent to actuators). If False (default),
                records the raw policy action.
                
        Example:
            >>> # Store each component separately (recommended for analysis)
            >>> recorder = RolloutRecorder(
            ...     recording_components=["gyros", "dof_pos", "dof_vel"],
            ...     separate_components=True,
            ... )
            >>> # Result: traj.keys() = ['gyros', 'dof_pos', 'dof_vel', 'actions', ...]
            
            >>> # Record final actuator commands instead of raw actions
            >>> recorder = RolloutRecorder(
            ...     recording_components=["dof_pos"],
            ...     action_as_actuator_command=True,
            ... )
            >>> # actions will contain: action + default_dof_pos
        """
        self.recording_components = recording_components or self.DEFAULT_RECORDING_COMPONENTS
        self.include_actions = include_actions
        self.include_rewards = include_rewards
        self.include_infos = include_infos
        self.custom_extractors = custom_extractors or {}
        self.separate_components = separate_components
        self.action_as_actuator_command = action_as_actuator_command
        
        # Cache for default_dof_pos (will be set on first record)
        self._default_dof_pos = None
        
        # Storage
        self.episodes: List[EpisodeData] = []
        self.current_episode: Optional[EpisodeData] = None
        self._recording = False
        
    def start_episode(self) -> None:
        """Start recording a new episode."""
        self.current_episode = EpisodeData()
        # Initialize custom data containers
        for key in self.custom_extractors.keys():
            self.current_episode.custom_data[key] = []
        # Initialize component data containers (for separate_components mode)
        if self.separate_components:
            for comp in self.recording_components:
                self.current_episode.component_data[comp] = []
        self._recording = True
        
    def end_episode(self) -> None:
        """End the current episode and save it."""
        if self.current_episode is not None:
            self.episodes.append(self.current_episode)
            self.current_episode = None
        self._recording = False
        
    def record(
        self,
        state: Any,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        info: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> None:
        """Record a single timestep.
        
        Args:
            state: The State object from the environment.
            action: Action taken (if include_actions is True).
            reward: Reward received (if include_rewards is True).
            info: Info dictionary (if include_infos is True).
            done: Whether the episode is done.
        """
        if not self._recording or self.current_episode is None:
            raise RuntimeError("Recording not started. Call start_episode() first.")
        
        # Extract and record state components
        if self.separate_components:
            # Store each component separately
            self._extract_components_separately(state)
        else:
            # Store all components concatenated into observations
            obs = self._extract_observation(state)
            self.current_episode.observations.append(obs)
        
        # Record action if requested
        if self.include_actions and action is not None:
            recorded_action = self._process_action(action, state)
            self.current_episode.actions.append(recorded_action)
        
        # Record reward if requested
        if self.include_rewards and reward is not None:
            self.current_episode.rewards.append(reward)
        
        # Record info if requested
        if self.include_infos and info is not None:
            # Make a shallow copy to avoid reference issues
            self.current_episode.infos.append(dict(info))
        
        # Record done
        self.current_episode.dones.append(done)
        
        # Run custom extractors
        for key, extractor in self.custom_extractors.items():
            try:
                value = extractor(state)
                self.current_episode.custom_data[key].append(value)
            except Exception as e:
                print(f"Warning: Custom extractor '{key}' failed: {e}")
                self.current_episode.custom_data[key].append(None)
    
    def _process_action(self, action: np.ndarray, state: Any) -> np.ndarray:
        """Process action for recording.
        
        Args:
            action: Raw action from policy.
            state: The State object (used to get default_dof_pos if needed).
            
        Returns:
            Processed action (either raw or as actuator command).
        """
        if not self.action_as_actuator_command:
            return action.copy()
        
        # Get default_dof_pos from state or its config
        default_dof_pos = self._get_default_dof_pos(state)
        
        if default_dof_pos is not None:
            # Return action + default_dof_pos (final actuator command)
            return action + default_dof_pos
        else:
            # Fallback to raw action if default_dof_pos not available
            return action.copy()
    
    def _get_default_dof_pos(self, state: Any) -> Optional[np.ndarray]:
        """Get default_dof_pos from state or cache.
        
        Args:
            state: The State object.
            
        Returns:
            default_dof_pos array or None if not available.
        """
        # Return cached value if available
        if self._default_dof_pos is not None:
            return self._default_dof_pos
        
        # Try to get from state's config
        if hasattr(state, 'cfg') and hasattr(state.cfg, 'control'):
            default_pos = getattr(state.cfg.control, 'default_dof_pos', None)
            if default_pos is not None:
                self._default_dof_pos = np.array(default_pos)
                return self._default_dof_pos
        
        # Try direct attribute on state
        if hasattr(state, 'default_dof_pos'):
            self._default_dof_pos = np.array(state.default_dof_pos)
            return self._default_dof_pos
        
        return None
    
    def set_default_dof_pos(self, default_dof_pos: np.ndarray) -> None:
        """Manually set the default_dof_pos for actuator command calculation.
        
        Use this if the recorder cannot automatically find default_dof_pos
        from the state object.
        
        Args:
            default_dof_pos: The default joint positions to add to actions.
        """
        self._default_dof_pos = np.array(default_dof_pos)
    
    def _extract_components_separately(self, state: Any) -> None:
        """Extract each component and store separately.
        
        Args:
            state: The State object from the environment.
        """
        for comp_name in self.recording_components:
            value = self._get_component_value(state, comp_name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    self.current_episode.component_data[comp_name].append(value.copy())
                elif isinstance(value, (list, tuple)):
                    self.current_episode.component_data[comp_name].append(np.array(value))
                elif isinstance(value, (int, float)):
                    self.current_episode.component_data[comp_name].append(np.array([value]))
                else:
                    self.current_episode.component_data[comp_name].append(value)
            else:
                self.current_episode.component_data[comp_name].append(None)
    
    def _extract_observation(self, state: Any) -> np.ndarray:
        """Extract observation from state based on recording components.
        
        Args:
            state: The State object from the environment.
            
        Returns:
            Flattened numpy array of extracted state components.
        """
        obs_parts = []
        for comp_name in self.recording_components:
            value = self._get_component_value(state, comp_name)
            if value is not None:
                if isinstance(value, np.ndarray):
                    obs_parts.append(value.flatten())
                elif isinstance(value, (list, tuple)):
                    obs_parts.append(np.array(value).flatten())
                elif isinstance(value, (int, float)):
                    obs_parts.append(np.array([value]))
                else:
                    # Skip non-numeric values
                    pass
        
        if obs_parts:
            return np.concatenate(obs_parts)
        return np.array([])
    
    def _get_component_value(self, state: Any, comp_name: str) -> Any:
        """Get a component value from state.
        
        Args:
            state: The State object.
            comp_name: Name of the component to extract.
            
        Returns:
            The component value, or None if not found.
        """
        # Try direct attribute access
        if hasattr(state, comp_name):
            return getattr(state, comp_name)
        
        # Try raw state
        if hasattr(state, 'raw') and hasattr(state.raw, comp_name):
            return getattr(state.raw, comp_name)
        
        # Try derived state
        if hasattr(state, 'derived') and hasattr(state.derived, comp_name):
            return getattr(state.derived, comp_name)
        
        # Try accurate state
        if hasattr(state, 'accurate') and hasattr(state.accurate, comp_name):
            return getattr(state.accurate, comp_name)
        
        # Try observable_data
        if hasattr(state, 'observable_data') and comp_name in state.observable_data:
            return state.observable_data[comp_name]
        
        return None
    
    @property
    def num_episodes(self) -> int:
        """Number of recorded episodes."""
        return len(self.episodes)
    
    @property
    def total_steps(self) -> int:
        """Total number of recorded steps across all episodes."""
        return sum(len(ep) for ep in self.episodes)
    
    def get_episode(self, idx: int) -> EpisodeData:
        """Get a specific episode's data."""
        return self.episodes[idx]
    
    def get_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Get recorded data as a list of trajectory dictionaries.
        
        Each trajectory is a dictionary containing data from a single episode.
        This format preserves episode boundaries and is useful for algorithms
        that process trajectories individually.
        
        Returns:
            List of dictionaries, where each dict contains:
            - If separate_components=False: 'observations' (concatenated), 'actions', 'rewards', 'dones'
            - If separate_components=True: each component name as a key (e.g., 'gyros', 'dof_pos'), 'actions', 'rewards', 'dones'
            - Plus any custom extractor keys
            
        Example:
            >>> trajectories = recorder.get_trajectories()
            >>> for traj in trajectories:
            ...     if 'gyros' in traj:  # separate_components=True
            ...         print(f"Gyros shape: {traj['gyros'].shape}")
            ...     print(f"Total reward: {traj['rewards'].sum()}")
        """
        trajectories = []
        for episode in self.episodes:
            # Don't include 'observations' if using separate_components mode
            traj = episode.to_dict(include_observations=not self.separate_components)
            trajectories.append(traj)
        return trajectories
    
    def get_all_data(self, as_trajectories: bool = False) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Get all recorded data.
        
        Args:
            as_trajectories: If True, returns a list of trajectory dicts.
                If False (default), returns concatenated data with episode_ids.
        
        Returns:
            If as_trajectories=True: List of trajectory dictionaries.
            If as_trajectories=False: Dictionary with concatenated data and episode_ids.
        """
        if not self.episodes:
            return [] if as_trajectories else {}
        
        # Return as list of trajectories if requested
        if as_trajectories:
            return self.get_trajectories()
        
        # Concatenate all episode data
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_episode_ids = []
        all_custom = {key: [] for key in self.custom_extractors.keys()}
        
        for ep_idx, episode in enumerate(self.episodes):
            ep_data = episode.to_dict()
            if len(ep_data["observations"]) > 0:
                all_obs.append(ep_data["observations"])
                all_episode_ids.extend([ep_idx] * len(ep_data["observations"]))
            if len(ep_data["actions"]) > 0:
                all_actions.append(ep_data["actions"])
            if len(ep_data["rewards"]) > 0:
                all_rewards.append(ep_data["rewards"])
            if len(ep_data["dones"]) > 0:
                all_dones.append(ep_data["dones"])
            
            for key in all_custom.keys():
                if key in ep_data and len(ep_data[key]) > 0:
                    all_custom[key].append(ep_data[key])
        
        result = {
            "observations": np.concatenate(all_obs) if all_obs else np.array([]),
            "episode_ids": np.array(all_episode_ids),
        }
        
        if all_actions:
            result["actions"] = np.concatenate(all_actions)
        if all_rewards:
            result["rewards"] = np.concatenate(all_rewards)
        if all_dones:
            result["dones"] = np.concatenate(all_dones)
        
        for key, values in all_custom.items():
            if values:
                result[key] = np.concatenate(values)

        
        return result
    
    def save(
        self,
        path: Union[str, Path],
        format: str = "npz",
        compress: bool = True,
        as_trajectories: bool = False,
    ) -> None:
        """Save recorded data to file.
        
        Args:
            path: Output file path.
            format: Output format ('npz', 'pkl', or 'hdf5').
            compress: Whether to compress the output (for npz format).
            as_trajectories: If True, saves data as a list of trajectory dicts.
                If False (default), saves concatenated data with episode_ids.
                Note: as_trajectories=True only works with 'pkl' format.
        """
        path = Path(path)
        
        if as_trajectories:
            # Save as list of trajectories (only pkl supports this well)
            if format != "pkl":
                print(f"Warning: as_trajectories=True works best with 'pkl' format. "
                      f"Switching from '{format}' to 'pkl'.")
                format = "pkl"
                path = path.with_suffix('.pkl')
            
            data = {
                "trajectories": self.get_trajectories(),
                "_metadata": {
                    "num_episodes": self.num_episodes,
                    "total_steps": self.total_steps,
                    "recording_components": self.recording_components,
                    "custom_extractors": list(self.custom_extractors.keys()),
                    "format": "trajectories",
                }
            }
        else:
            # Save as concatenated data
            data = self.get_all_data(as_trajectories=False)
            data["_metadata"] = np.array([{
                "num_episodes": self.num_episodes,
                "total_steps": self.total_steps,
                "recording_components": self.recording_components,
                "custom_extractors": list(self.custom_extractors.keys()),
                "format": "concatenated",
            }])

        if format == "npz":
            if compress:
                np.savez_compressed(path, **data)
            else:
                np.savez(path, **data)
        elif format == "pkl":
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "hdf5":
            if as_trajectories:
                raise ValueError("HDF5 format does not support as_trajectories=True. Use 'pkl' instead.")
            try:
                import h5py
                with h5py.File(path, 'w') as f:
                    for key, value in data.items():
                        if key == "_metadata":
                            f.attrs["metadata"] = str(value[0])
                        else:
                            f.create_dataset(key, data=value)
            except ImportError:
                raise ImportError("h5py is required for HDF5 format. Install with: pip install h5py")
        else:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'pkl', or 'hdf5'.")
        
        format_str = "trajectories" if as_trajectories else "concatenated"
        print(f"Saved {self.num_episodes} episodes ({self.total_steps} steps) as {format_str} to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Union[Dict[str, np.ndarray], Dict[str, Any]]:
        """Load recorded data from file.
        
        Args:
            path: Input file path.
            
        Returns:
            If saved with as_trajectories=True:
                Dict with 'trajectories' (list of episode dicts) and '_metadata'.
            If saved with as_trajectories=False:
                Dict with concatenated arrays ('observations', 'actions', etc.)
                and 'episode_ids' to identify episode boundaries.
                
        Example:
            >>> # Load concatenated data
            >>> data = RolloutRecorder.load("rollouts.npz")
            >>> obs = data["observations"]  # Shape: (total_steps, obs_dim)
            >>> 
            >>> # Load trajectory data
            >>> data = RolloutRecorder.load("rollouts.pkl")
            >>> if "trajectories" in data:
            ...     for traj in data["trajectories"]:
            ...         print(f"Episode reward: {traj['rewards'].sum()}")
        """
        path = Path(path)
        
        if path.suffix == '.npz':
            data = dict(np.load(path, allow_pickle=True))
        elif path.suffix == '.pkl':
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif path.suffix in ['.h5', '.hdf5']:
            import h5py
            data = {}
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
                if "metadata" in f.attrs:
                    data["_metadata"] = eval(f.attrs["metadata"])
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        
        return data
    
    def clear(self) -> None:
        """Clear all recorded data."""
        self.episodes = []
        self.current_episode = None
        self._recording = False
        self._default_dof_pos = None  # Reset cached default_dof_pos


class StateSnapshot:
    """A lightweight snapshot of state data for recording.
    
    Use this when you need to record specific state values without
    keeping references to the full state object.
    """
    
    def __init__(self, state: Any, components: Optional[List[str]] = None):
        """Create a snapshot of state data.
        
        Args:
            state: The State object to snapshot.
            components: List of component names to capture.
                If None, captures all available components.
        """
        self.data = {}
        
        if components is None:
            components = RolloutRecorder.DEFAULT_RECORDING_COMPONENTS
        
        recorder = RolloutRecorder(recording_components=[])
        for comp in components:
            value = recorder._get_component_value(state, comp)
            if value is not None:
                if isinstance(value, np.ndarray):
                    self.data[comp] = value.copy()
                else:
                    self.data[comp] = value
    
    def __getattr__(self, name: str) -> Any:
        if name in ['data']:
            return super().__getattribute__(name)
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"StateSnapshot has no attribute '{name}'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return dict(self.data)
    
    def to_array(self) -> np.ndarray:
        """Convert snapshot to flattened array."""
        parts = []
        for value in self.data.values():
            if isinstance(value, np.ndarray):
                parts.append(value.flatten())
            elif isinstance(value, (int, float)):
                parts.append(np.array([value]))
        return np.concatenate(parts) if parts else np.array([])
