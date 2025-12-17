"""
Ray-based Vectorized Environment for MetaMachine

This module provides a vectorized environment wrapper that uses Ray for
parallel environment execution. The wrapper is compatible with the rsl_rl
VecEnv interface for easy integration with reinforcement learning training.

The implementation uses Ray actors to run multiple MetaMachine environments
in parallel, enabling efficient data collection for training.

Example:
    >>> from metamachine.environments.vec_env import RayVecMetaMachine
    >>> from metamachine.environments.configs.config_registry import ConfigRegistry
    >>> 
    >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
    >>> vec_env = RayVecMetaMachine(cfg, num_envs=8)
    >>> 
    >>> obs = vec_env.reset()
    >>> for _ in range(100):
    ...     actions = policy(obs)
    ...     obs, rewards, dones, infos = vec_env.step(actions)
    >>> vec_env.close()

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import torch, but make it optional for non-GPU setups
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import ray, but make it optional
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class StateSnapshot:
    """Serializable snapshot of environment state for parallel recording.
    
    This class captures all relevant state information in a serializable format
    that can be transferred between Ray actors. Unlike the full State object,
    this does not contain any MuJoCo objects or other non-serializable data.
    
    Attributes:
        pos_world: World position of the robot base.
        quat: Orientation quaternion of the robot base.
        vel_world: Linear velocity in world frame.
        vel_body: Linear velocity in body frame.
        ang_vel_world: Angular velocity in world frame.
        ang_vel_body: Angular velocity in body frame.
        dof_pos: Joint positions.
        dof_vel: Joint velocities.
        projected_gravity: Gravity vector projected into body frame.
        projected_gravities: Per-module gravity projections.
        gyros: Per-module gyroscope readings.
        accs: Per-module accelerometer readings.
        quats: Per-module quaternions.
        height: Robot height.
        heading: Robot heading.
        speed: Robot speed.
        commands: Command values.
        last_action: Last action taken.
        accurate_vel_world: Accurate world velocity (ground truth).
        accurate_pos_world: Accurate world position (ground truth).
        contact_floor_balls: Ball contacts with floor.
        contact_floor_geoms: Geometry contacts with floor.
    """
    # Core state
    pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Joint state
    dof_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))
    dof_vel: np.ndarray = field(default_factory=lambda: np.zeros(1))
    
    # Derived state
    projected_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    projected_gravities: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    height: np.ndarray = field(default_factory=lambda: np.zeros(1))
    heading: np.ndarray = field(default_factory=lambda: np.zeros(1))
    speed: np.ndarray = field(default_factory=lambda: np.zeros(1))
    
    # Sensor data
    gyros: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    accs: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    quats: List[np.ndarray] = field(default_factory=list)
    
    # Commands and actions
    commands: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(1))
    
    # Accurate/ground truth state
    accurate_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accurate_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accurate_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accurate_ang_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Contact information
    contact_floor_balls: List[int] = field(default_factory=list)
    contact_floor_geoms: List[int] = field(default_factory=list)
    contact_floor_socks: List[int] = field(default_factory=list)
    num_jointfloor_contact: int = 0
    
    # Configuration (stored for reference)
    num_modules: int = 1
    default_dof_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))
    
    @classmethod
    def from_state(cls, state: Any) -> "StateSnapshot":
        """Create a StateSnapshot from a full State object.
        
        Args:
            state: The State object from MetaMachine environment.
            
        Returns:
            StateSnapshot with all data copied from state.
        """
        snapshot = cls()
        
        # Copy core state from raw state
        if hasattr(state, 'raw'):
            raw = state.raw
            snapshot.pos_world = np.array(raw.pos_world).copy()
            snapshot.quat = np.array(raw.quat).copy()
            snapshot.vel_world = np.array(raw.vel_world).copy()
            snapshot.vel_body = np.array(raw.vel_body).copy()
            snapshot.ang_vel_world = np.array(raw.ang_vel_world).copy()
            snapshot.ang_vel_body = np.array(raw.ang_vel_body).copy()
            snapshot.dof_pos = np.array(raw.dof_pos).copy()
            snapshot.dof_vel = np.array(raw.dof_vel).copy()
            
            if raw.gyros is not None:
                snapshot.gyros = np.array(raw.gyros).copy()
            if raw.accs is not None:
                snapshot.accs = np.array(raw.accs).copy()
            if raw.quats is not None:
                snapshot.quats = [np.array(q).copy() for q in raw.quats]
            
            snapshot.contact_floor_balls = list(raw.contact_floor_balls)
            snapshot.contact_floor_geoms = list(raw.contact_floor_geoms)
            snapshot.contact_floor_socks = list(raw.contact_floor_socks)
        
        # Copy derived state
        if hasattr(state, 'derived'):
            derived = state.derived
            snapshot.projected_gravity = np.array(derived.projected_gravity).copy()
            if derived.projected_gravities is not None:
                snapshot.projected_gravities = np.array(derived.projected_gravities).copy()
            snapshot.height = np.array(derived.height).copy()
            snapshot.heading = np.array(derived.heading).copy()
            snapshot.speed = np.array(derived.speed).copy()
        
        # Copy accurate state (ground truth)
        if hasattr(state, 'accurate'):
            accurate = state.accurate
            if accurate.vel_world is not None:
                snapshot.accurate_vel_world = np.array(accurate.vel_world).copy()
            if accurate.pos_world is not None:
                snapshot.accurate_pos_world = np.array(accurate.pos_world).copy()
            if accurate.vel_body is not None:
                snapshot.accurate_vel_body = np.array(accurate.vel_body).copy()
            if accurate.ang_vel_body is not None:
                snapshot.accurate_ang_vel_body = np.array(accurate.ang_vel_body).copy()
        
        # Copy commands
        if hasattr(state, 'commands'):
            snapshot.commands = np.array(state.commands).copy()
        
        # Copy last action
        if hasattr(state, 'action_history') and hasattr(state.action_history, 'last_action'):
            snapshot.last_action = np.array(state.action_history.last_action).copy()
        
        # Copy contact info from observable_data
        if hasattr(state, 'observable_data'):
            obs_data = state.observable_data
            if 'num_jointfloor_contact' in obs_data:
                snapshot.num_jointfloor_contact = obs_data['num_jointfloor_contact']
        
        # Store number of modules
        if hasattr(state, 'num_act'):
            snapshot.num_modules = state.num_act
        
        # Copy default_dof_pos from config
        if hasattr(state, 'cfg') and hasattr(state.cfg, 'control'):
            default_pos = getattr(state.cfg.control, 'default_dof_pos', None)
            if default_pos is not None:
                snapshot.default_dof_pos = np.array(default_pos).copy()
        
        return snapshot
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary format.
        
        Returns:
            Dictionary with all state data.
        """
        return {
            'pos_world': self.pos_world,
            'quat': self.quat,
            'vel_world': self.vel_world,
            'vel_body': self.vel_body,
            'ang_vel_world': self.ang_vel_world,
            'ang_vel_body': self.ang_vel_body,
            'dof_pos': self.dof_pos,
            'dof_vel': self.dof_vel,
            'projected_gravity': self.projected_gravity,
            'projected_gravities': self.projected_gravities,
            'height': self.height,
            'heading': self.heading,
            'speed': self.speed,
            'gyros': self.gyros,
            'accs': self.accs,
            'quats': self.quats,
            'commands': self.commands,
            'last_action': self.last_action,
            'accurate_vel_world': self.accurate_vel_world,
            'accurate_pos_world': self.accurate_pos_world,
            'accurate_vel_body': self.accurate_vel_body,
            'accurate_ang_vel_body': self.accurate_ang_vel_body,
            'contact_floor_balls': self.contact_floor_balls,
            'contact_floor_geoms': self.contact_floor_geoms,
            'contact_floor_socks': self.contact_floor_socks,
            'num_jointfloor_contact': self.num_jointfloor_contact,
            'num_modules': self.num_modules,
            'default_dof_pos': self.default_dof_pos,
        }


class VecEnv(ABC):
    """Abstract base class for vectorized environments (compatible with rsl_rl).
    
    This class defines the interface for vectorized environments that can be
    used with rsl_rl and other RL training frameworks.
    
    Attributes:
        num_envs: Number of parallel environments.
        num_obs: Dimension of observations.
        num_privileged_obs: Dimension of privileged observations (optional).
        num_actions: Dimension of action space.
        max_episode_length: Maximum episode length.
        device: Device for tensor operations.
    """
    
    num_envs: int
    num_obs: int
    num_privileged_obs: Optional[int]
    num_actions: int
    max_episode_length: int
    device: str
    
    @abstractmethod
    def reset(self) -> Tuple[Any, Optional[Any]]:
        """Reset all environments.
        
        Returns:
            Tuple of (observations, privileged_observations).
            privileged_observations may be None if not available.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, actions: Any) -> Tuple[Any, Any, Any, Any, Any]:
        """Step all environments with given actions.
        
        Args:
            actions: Actions tensor of shape (num_envs, num_actions).
            
        Returns:
            Tuple of (obs, privileged_obs, rewards, dones, infos).
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_observations(self) -> Any:
        """Get current observations from all environments.
        
        Returns:
            Observations tensor of shape (num_envs, num_obs).
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """Close all environments and release resources."""
        pass


def _create_remote_metamachine_class():
    """Factory function to create RemoteMetaMachine class.
    
    This is needed because the @ray.remote decorator requires ray to be imported.
    """
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray is required for vectorized environments. "
            "Install with: pip install ray"
        )
    
    @ray.remote
    class RemoteMetaMachine:
        """Ray remote MetaMachine environment wrapper.
        
        This class wraps a single MetaMachine environment as a Ray actor,
        enabling parallel execution of multiple environments.
        """
        
        def __init__(self, cfg, env_id: int):
            """Initialize remote environment.
            
            Args:
                cfg: Environment configuration (OmegaConf object).
                env_id: Environment ID for identification.
            """
            # Set EGL for headless rendering
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
            
            # Deep copy config to avoid shared state issues
            self.cfg = copy.deepcopy(cfg)
            self.env_id = env_id
            
            # Disable rendering for remote environments
            self.cfg.simulation.render_mode = "none"
            if hasattr(self.cfg, 'logging'):
                self.cfg.logging.create_log_dir = False
            
            # Import here to avoid circular imports
            from metamachine.environments.env_sim import MetaMachine
            
            # Create the environment
            self.env = MetaMachine(self.cfg)
            
            # Store environment info
            self.num_obs = self.env.observation_space.shape[0]
            self.num_actions = self.env.action_space.shape[0]
            self.max_episode_length = getattr(
                self.cfg.task.termination_conditions, 
                'max_episode_steps', 
                1000
            )
        
        def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Reset environment and return observation.
            
            Args:
                seed: Optional random seed.
                
            Returns:
                Tuple of (observation, info).
            """
            obs, info = self.env.reset(seed=seed)
            return obs.astype(np.float32), info
        
        def reset_with_state(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any], StateSnapshot]:
            """Reset environment and return observation with state snapshot.
            
            Args:
                seed: Optional random seed.
                
            Returns:
                Tuple of (observation, info, state_snapshot).
            """
            obs, info = self.env.reset(seed=seed)
            state_snapshot = StateSnapshot.from_state(self.env.state)
            return obs.astype(np.float32), info, state_snapshot
        
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            """Step environment with given action.
            
            Args:
                action: Action array.
                
            Returns:
                Tuple of (obs, reward, done, truncated, info).
            """
            obs, reward, done, truncated, info = self.env.step(action)
            return obs.astype(np.float32), float(reward), done, truncated, info
        
        def step_with_state(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any], StateSnapshot]:
            """Step environment and return state snapshot.
            
            Args:
                action: Action array.
                
            Returns:
                Tuple of (obs, reward, done, truncated, info, state_snapshot).
            """
            obs, reward, done, truncated, info = self.env.step(action)
            state_snapshot = StateSnapshot.from_state(self.env.state)
            return obs.astype(np.float32), float(reward), done, truncated, info, state_snapshot
        
        def step_with_auto_reset(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
            """Step environment with automatic reset on episode end.
            
            Args:
                action: Action array.
                
            Returns:
                Tuple of (obs, reward, done, info).
                If done, obs is from the reset environment.
            """
            obs, reward, done, truncated, info = self.env.step(action)
            episode_done = done or truncated
            
            if episode_done:
                obs, _ = self.env.reset()
            
            return obs.astype(np.float32), float(reward), episode_done, info
        
        def step_with_auto_reset_and_state(
            self, action: np.ndarray
        ) -> Tuple[np.ndarray, float, bool, Dict[str, Any], StateSnapshot]:
            """Step environment with automatic reset and return state snapshot.
            
            The state snapshot is captured BEFORE auto-reset (represents the state
            that produced the transition). This is important for recording rollouts.
            
            Args:
                action: Action array.
                
            Returns:
                Tuple of (obs, reward, done, info, state_snapshot).
                state_snapshot is from BEFORE reset if episode ended.
                obs is from AFTER reset if episode ended.
            """
            obs, reward, done, truncated, info = self.env.step(action)
            episode_done = done or truncated
            
            # Capture state snapshot BEFORE reset
            state_snapshot = StateSnapshot.from_state(self.env.state)
            
            if episode_done:
                obs, _ = self.env.reset()
            
            return obs.astype(np.float32), float(reward), episode_done, info, state_snapshot
        
        def get_observation(self) -> np.ndarray:
            """Get current observation without stepping.
            
            Returns:
                Current observation array.
            """
            return self.env.state.get_observation(insert=False).astype(np.float32)
        
        def get_state_snapshot(self) -> StateSnapshot:
            """Get current state as a serializable snapshot.
            
            Returns:
                StateSnapshot containing all relevant state data.
            """
            return StateSnapshot.from_state(self.env.state)
        
        def get_env_info(self) -> Dict[str, Any]:
            """Get environment information.
            
            Returns:
                Dictionary with environment info.
            """
            return {
                'num_obs': self.num_obs,
                'num_actions': self.num_actions,
                'max_episode_length': self.max_episode_length,
                'env_id': self.env_id,
                'num_modules': self.env.state.num_act if hasattr(self.env, 'state') else self.num_actions,
            }
        
        def close(self) -> None:
            """Close the environment."""
            self.env.close()
    
    return RemoteMetaMachine


class RayVecMetaMachine(VecEnv):
    """Ray-based vectorized MetaMachine environment for multiprocessing.
    
    This class provides a vectorized environment wrapper that uses Ray for
    parallel execution of multiple MetaMachine environments. It implements
    the VecEnv interface from rsl_rl for easy integration with RL training.
    
    Attributes:
        num_envs: Number of parallel environments.
        device: Device for tensor operations.
        cfg: Environment configuration.
        envs: List of Ray actor references.
        num_obs: Dimension of observations.
        num_actions: Dimension of action space.
        max_episode_length: Maximum episode length.
    """
    
    def __init__(
        self,
        cfg,
        num_envs: int = 1,
        device: str = "cuda:0",
        num_cpus_per_env: float = 0.25,
        num_gpus_per_env: float = 0.0,
        ray_temp_dir: Optional[str] = None,
        use_torch: bool = True,
    ):
        """Initialize Ray-based vectorized environment.
        
        Args:
            cfg: Environment configuration (OmegaConf object).
            num_envs: Number of parallel environments.
            device: PyTorch device for tensors (e.g., "cuda:0", "cpu").
            num_cpus_per_env: CPU resources per environment actor. Use fractional
                values (e.g., 0.25) to allow multiple actors to share CPUs when
                running many environments. Default is 0.25, allowing 4 envs per CPU.
                If you get "resource request cannot be scheduled" errors, try
                reducing this value (e.g., 0.1 for 10 envs per CPU).
            num_gpus_per_env: GPU resources per environment actor.
            ray_temp_dir: Temporary directory for Ray.
            use_torch: Whether to return torch tensors (True) or numpy arrays (False).
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is required for RayVecMetaMachine. "
                "Install with: pip install ray"
            )
        
        self.use_torch = use_torch and TORCH_AVAILABLE
        if use_torch and not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Using numpy arrays instead.")
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            init_kwargs = {"ignore_reinit_error": True}
            if ray_temp_dir is not None:
                init_kwargs["_temp_dir"] = ray_temp_dir
            ray.init(**init_kwargs)
        
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        self.num_cpus_per_env = num_cpus_per_env
        self.num_gpus_per_env = num_gpus_per_env
        
        # Create remote environment class
        RemoteMetaMachine = _create_remote_metamachine_class()
        
        # Create remote environment actors with resource allocation
        self.envs = []
        for i in range(num_envs):
            # Configure Ray actor with resource requirements
            RemoteEnvClass = RemoteMetaMachine.options(
                num_cpus=num_cpus_per_env,
                num_gpus=num_gpus_per_env
            )
            env_actor = RemoteEnvClass.remote(cfg, i)
            self.envs.append(env_actor)
        
        # Get environment info from first environment
        env_info = ray.get(self.envs[0].get_env_info.remote())
        self.num_obs = env_info['num_obs']
        self.num_actions = env_info['num_actions']
        self.max_episode_length = env_info['max_episode_length']
        self.num_modules = env_info.get('num_modules', self.num_actions)
        self.num_privileged_obs = None  # Can be extended if needed
        
        # Initialize tracking
        self._obs_buffer = None
        self._state_buffer: List[Optional[StateSnapshot]] = [None] * num_envs
        self._episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self._episode_rewards = np.zeros(num_envs, dtype=np.float32)
        
        # Initialize environments
        self._initialized = False
    
    def _to_tensor(self, data: np.ndarray) -> Any:
        """Convert numpy array to tensor if using torch.
        
        Args:
            data: Numpy array to convert.
            
        Returns:
            Tensor if use_torch is True, otherwise numpy array.
        """
        if self.use_torch:
            return torch.tensor(data, device=self.device, dtype=torch.float32)
        return data
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert tensor to numpy array.
        
        Args:
            data: Tensor or numpy array.
            
        Returns:
            Numpy array.
        """
        if self.use_torch and torch.is_tensor(data):
            return data.cpu().numpy()
        return np.asarray(data)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Optional[Any]]:
        """Reset all environments and return observations.
        
        Args:
            seed: Optional base seed for reproducibility.
                Each environment will use seed + env_id.
            
        Returns:
            Tuple of (observations, privileged_observations).
            Observations shape: (num_envs, num_obs).
        """
        # Reset all environments in parallel
        if seed is not None:
            reset_futures = [
                env.reset.remote(seed=seed + i) 
                for i, env in enumerate(self.envs)
            ]
        else:
            reset_futures = [env.reset.remote() for env in self.envs]
        
        results = ray.get(reset_futures)
        obs_list, _ = zip(*results)
        
        # Stack observations
        obs_np = np.stack(obs_list, axis=0)
        self._obs_buffer = obs_np.copy()
        
        # Reset tracking
        self._episode_lengths.fill(0)
        self._episode_rewards.fill(0.0)
        self._state_buffer = [None] * self.num_envs
        self._initialized = True
        
        obs = self._to_tensor(obs_np)
        return obs, None  # No privileged observations for now
    
    def reset_with_states(self, seed: Optional[int] = None) -> Tuple[Any, Optional[Any], List[StateSnapshot]]:
        """Reset all environments and return observations with state snapshots.
        
        Args:
            seed: Optional base seed for reproducibility.
            
        Returns:
            Tuple of (observations, privileged_observations, states).
        """
        # Reset all environments in parallel with state snapshots
        if seed is not None:
            reset_futures = [
                env.reset_with_state.remote(seed=seed + i) 
                for i, env in enumerate(self.envs)
            ]
        else:
            reset_futures = [env.reset_with_state.remote() for env in self.envs]
        
        results = ray.get(reset_futures)
        obs_list, _, state_list = zip(*results)
        
        # Stack observations
        obs_np = np.stack(obs_list, axis=0)
        self._obs_buffer = obs_np.copy()
        self._state_buffer = list(state_list)
        
        # Reset tracking
        self._episode_lengths.fill(0)
        self._episode_rewards.fill(0.0)
        self._initialized = True
        
        obs = self._to_tensor(obs_np)
        return obs, None, list(state_list)
    
    def step(self, actions: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        """Step all environments with given actions.
        
        Args:
            actions: Actions tensor/array of shape (num_envs, num_actions).
            
        Returns:
            Tuple of (obs, privileged_obs, rewards, dones, infos).
            - obs: (num_envs, num_obs)
            - privileged_obs: None (not implemented)
            - rewards: (num_envs,)
            - dones: (num_envs,) boolean
            - infos: Dictionary with additional info
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert actions to numpy
        actions_np = self._to_numpy(actions)
        
        # Step all environments in parallel with auto-reset
        step_futures = [
            self.envs[i].step_with_auto_reset.remote(actions_np[i])
            for i in range(self.num_envs)
        ]
        results = ray.get(step_futures)
        
        # Unpack results
        obs_list, reward_list, done_list, info_list = zip(*results)
        
        # Convert to arrays
        obs_np = np.stack(obs_list, axis=0)
        rewards_np = np.array(reward_list, dtype=np.float32)
        dones_np = np.array(done_list, dtype=bool)
        
        # Update tracking
        self._obs_buffer = obs_np.copy()
        self._episode_lengths += 1
        self._episode_rewards += rewards_np
        
        # Build info dictionary
        infos = {
            "observations": {
                "critic": self._to_tensor(obs_np),  # Same as actor obs for now
            },
            "episode_lengths": self._episode_lengths.copy(),
            "episode_rewards": self._episode_rewards.copy(),
            "original_infos": info_list,
        }
        
        # Add episode info for completed episodes
        if dones_np.any():
            infos["episode"] = {
                "r": self._episode_rewards[dones_np].copy(),
                "l": self._episode_lengths[dones_np].copy(),
            }
            # Reset tracking for done environments
            self._episode_lengths[dones_np] = 0
            self._episode_rewards[dones_np] = 0.0
        
        # Convert outputs
        obs = self._to_tensor(obs_np)
        privileged_obs = None
        rewards = self._to_tensor(rewards_np)
        dones = self._to_tensor(dones_np.astype(np.float32))
        
        return obs, privileged_obs, rewards, dones, infos
    
    def step_with_states(
        self, actions: Any
    ) -> Tuple[Any, Any, Any, Any, Dict[str, Any], List[StateSnapshot]]:
        """Step all environments and return state snapshots.
        
        This method is useful for recording rollouts where you need access to
        the full state information for each environment.
        
        Args:
            actions: Actions tensor/array of shape (num_envs, num_actions).
            
        Returns:
            Tuple of (obs, privileged_obs, rewards, dones, infos, states).
            states is a list of StateSnapshot objects from each environment.
        """
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert actions to numpy
        actions_np = self._to_numpy(actions)
        
        # Step all environments in parallel with auto-reset and state capture
        step_futures = [
            self.envs[i].step_with_auto_reset_and_state.remote(actions_np[i])
            for i in range(self.num_envs)
        ]
        results = ray.get(step_futures)
        
        # Unpack results
        obs_list, reward_list, done_list, info_list, state_list = zip(*results)
        
        # Convert to arrays
        obs_np = np.stack(obs_list, axis=0)
        rewards_np = np.array(reward_list, dtype=np.float32)
        dones_np = np.array(done_list, dtype=bool)
        
        # Update tracking
        self._obs_buffer = obs_np.copy()
        self._state_buffer = list(state_list)
        self._episode_lengths += 1
        self._episode_rewards += rewards_np
        
        # Build info dictionary
        infos = {
            "observations": {
                "critic": self._to_tensor(obs_np),
            },
            "episode_lengths": self._episode_lengths.copy(),
            "episode_rewards": self._episode_rewards.copy(),
            "original_infos": info_list,
        }
        
        # Add episode info for completed episodes
        if dones_np.any():
            infos["episode"] = {
                "r": self._episode_rewards[dones_np].copy(),
                "l": self._episode_lengths[dones_np].copy(),
            }
            self._episode_lengths[dones_np] = 0
            self._episode_rewards[dones_np] = 0.0
        
        # Convert outputs
        obs = self._to_tensor(obs_np)
        privileged_obs = None
        rewards = self._to_tensor(rewards_np)
        dones = self._to_tensor(dones_np.astype(np.float32))
        
        return obs, privileged_obs, rewards, dones, infos, list(state_list)
    
    def get_observations(self) -> Any:
        """Get current observations from all environments.
        
        Returns:
            Observations tensor/array of shape (num_envs, num_obs).
        """
        if self._obs_buffer is None:
            # Get observations from all environments in parallel
            obs_futures = [env.get_observation.remote() for env in self.envs]
            obs_list = ray.get(obs_futures)
            obs_np = np.stack(obs_list, axis=0)
            self._obs_buffer = obs_np.copy()
        
        return self._to_tensor(self._obs_buffer)
    
    def get_states(self) -> List[StateSnapshot]:
        """Get current state snapshots from all environments.
        
        Returns:
            List of StateSnapshot objects from each environment.
        """
        state_futures = [env.get_state_snapshot.remote() for env in self.envs]
        states = ray.get(state_futures)
        self._state_buffer = list(states)
        return list(states)
    
    def get_privileged_observations(self) -> Optional[Any]:
        """Get privileged observations (not implemented).
        
        Returns:
            None (privileged observations not implemented).
        """
        return None
    
    def close(self) -> None:
        """Close all remote environments and cleanup."""
        # Close each environment
        close_futures = [env.close.remote() for env in self.envs]
        try:
            ray.get(close_futures, timeout=5.0)
        except Exception:
            pass
        
        # Terminate remote actors
        for env in self.envs:
            try:
                ray.kill(env)
            except Exception:
                pass
        
        self.envs = []
        self._initialized = False
