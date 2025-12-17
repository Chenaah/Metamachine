"""
Batch Rollout Recording Example

This example demonstrates how to record robot rollouts using the vectorized
MetaMachine environment. By running multiple environments in parallel using
Ray, this approach significantly speeds up rollout collection compared to
the sequential version.

Key Features:
- Parallel environment execution using RayVecMetaMachine
- Batch policy inference for improved throughput
- Recording full state information from each environment
- Compatible with rsl_rl VecEnv interface

Use cases:
1. Fast data collection for imitation learning
2. Parallel evaluation of policies
3. Large-scale rollout recording for analysis

Configuration:
    Modify the global variables at the top of this file to customize behavior:
    
    # Number of parallel environments:
    NUM_ENVS = 8
    
    # To use a different model from registry:
    MODEL = "your_model_name"
    
    # To use a local policy file:
    POLICY_PATH = "./path/to/your/policy.pkl"
    MODEL = None  # Set this to None when using POLICY_PATH
    
    # To record episodes:
    NUM_EPISODES = 100  # Total episodes to collect
    
    # To save in different format:
    OUTPUT_FORMAT = "pkl"
    OUTPUT = "batch_rollouts.pkl"

Usage:
    # Simply run the script after configuring the global variables
    python batch_record_rollout.py

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
"""

import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from metamachine.utils.checkpoint_manager import CheckpointManager
from metamachine.utils.rollout_recorder import RolloutRecorder

# ============================================================================
# Configuration - modify these variables to change behavior
# ============================================================================

# Model loading options
MODEL = ["three_modules_run_policy", "quadruped_run_policy"][1]  # Name of registered model
POLICY_PATH = None  # Path to local policy file (.pkl) - set MODEL to None if using this

# Environment options
CONFIG = ["example_three_modules", "basic_quadruped"][1]  # Environment configuration name
N_MODULES = 5  # Number of modules in the robot

# Parallelization options
NUM_ENVS = 50  # Number of parallel environments
NUM_CPUS_PER_ENV = 0.2  # CPU resources per environment (fractional allows oversubscription)
NUM_GPUS_PER_ENV = 0.0  # GPU resources per environment
DEVICE = "cpu"  # Device for policy inference ("cuda:0" or "cpu")

# Recording options
NUM_EPISODES = 1000  # Total number of episodes to record
MAX_STEPS = 1000  # Maximum steps per episode
OUTPUT = "batch_rollouts.pkl"  # Output file path
OUTPUT_FORMAT = "pkl"  # Output format: "npz", "pkl", or "hdf5"
RECORDING_COMPONENTS = ["accurate_vel_world"]  # State components to record

# Simulation options
SEED = 42  # Random seed for reproducibility
VERBOSE = True  # Print detailed information during recording

# Ray options
RAY_TEMP_DIR = None  # Temporary directory for Ray (None for default)

# ============================================================================
# Lazy imports of optional dependencies
# ============================================================================
CrossQ = None
ConfigRegistry = None
MetaMachine = None
RayVecMetaMachine = None
StateSnapshot = None


def create_custom_extractors(n_modules: int = N_MODULES) -> Dict[str, Callable]:
    """Create custom data extractors for recording additional information.
    
    These extractors work with StateSnapshot objects from the vectorized environment.
    
    Args:
        n_modules: Number of modules in the robot.
    
    Returns:
        Dictionary of extractor functions.
    """
    extractors = {}
    
    for i in range(n_modules):
        # Create extractor for each module
        def module_extractor(state, idx=i):
            """Extract per-module observation data."""
            # Get data from StateSnapshot
            proj_grav = state.projected_gravities[idx] if len(state.projected_gravities) > idx else np.zeros(3)
            gyro = state.gyros[idx] if len(state.gyros) > idx else np.zeros(3)
            dof_pos = state.dof_pos[idx:idx+1] if len(state.dof_pos) > idx else np.zeros(1)
            dof_vel = state.dof_vel[idx:idx+1] if len(state.dof_vel) > idx else np.zeros(1)
            
            return np.concatenate([
                np.asarray(proj_grav).flatten(),
                np.asarray(gyro).flatten(),
                np.asarray(dof_pos).flatten(),
                np.asarray(dof_vel).flatten()
            ])
        
        extractors[f"module{i}"] = module_extractor
    
    return extractors


class BatchRolloutRecorder:
    """Batch rollout recorder for vectorized environments.
    
    This class manages multiple RolloutRecorder instances, one for each
    parallel environment, and handles the coordination of recording
    across all environments.
    
    The recorder works with StateSnapshot objects from RayVecMetaMachine,
    which contain the full state information from each remote environment.
    """
    
    def __init__(
        self,
        num_envs: int,
        recording_components: Optional[List[str]] = None,
        separate_components: bool = True,
        include_actions: bool = True,
        include_rewards: bool = True,
        include_infos: bool = False,
        include_env_obs: bool = True,
        custom_extractors: Optional[Dict[str, Callable]] = None,
        action_as_actuator_command: bool = True,
    ):
        """Initialize batch rollout recorder.
        
        Args:
            num_envs: Number of parallel environments.
            recording_components: List of state component names to record.
            separate_components: Store each component separately.
            include_actions: Whether to record actions.
            include_rewards: Whether to record rewards.
            include_infos: Whether to record info dicts.
            include_env_obs: Whether to record environment observations.
            custom_extractors: Custom data extraction functions.
            action_as_actuator_command: Record final actuator command.
        """
        self.num_envs = num_envs
        self.recording_components = recording_components
        self.custom_extractors = custom_extractors or {}
        self.include_env_obs = include_env_obs
        
        # Create a recorder for each environment
        # Pass custom_extractors to each recorder so they can extract data from StateSnapshotWrapper
        self.recorders = [
            RolloutRecorder(
                recording_components=recording_components,
                separate_components=separate_components,
                include_actions=include_actions,
                include_rewards=include_rewards,
                include_infos=include_infos,
                include_env_obs=include_env_obs,
                custom_extractors=self.custom_extractors,  # Pass the actual extractors
                action_as_actuator_command=action_as_actuator_command,
            )
            for _ in range(num_envs)
        ]
        
        # Track episode counts
        self.episode_counts = np.zeros(num_envs, dtype=np.int32)
        self.total_episodes = 0
        self.total_steps = 0
        
    def start_episodes(self) -> None:
        """Start recording episodes for all environments."""
        for recorder in self.recorders:
            recorder.start_episode()
    
    def record_batch(
        self,
        states: List[Any],
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: Optional[List[Dict[str, Any]]] = None,
        env_obs: Optional[np.ndarray] = None,
    ) -> int:
        """Record a batch of timesteps from all environments.
        
        Args:
            states: List of StateSnapshot objects from each environment.
            actions: Actions array of shape (num_envs, num_actions).
            rewards: Rewards array of shape (num_envs,).
            dones: Done flags array of shape (num_envs,).
            infos: Optional list of info dicts from each environment.
            env_obs: Optional environment observations array.
            
        Returns:
            Number of episodes completed in this batch.
        """
        completed = 0
        
        for i in range(self.num_envs):
            state = states[i] if states else None
            action = actions[i] if actions is not None else None
            reward = float(rewards[i]) if rewards is not None else None
            done = bool(dones[i])
            info = infos[i] if infos else None
            
            # Get the environment observation for this environment if provided
            obs_i = env_obs[i] if env_obs is not None else None
            
            # For StateSnapshot, we need to handle recording differently
            # The RolloutRecorder expects a State object with specific attributes
            # We create a wrapper that provides the same interface
            if state is not None:
                state_wrapper = StateSnapshotWrapper(state, self.custom_extractors, env_obs=obs_i)
            else:
                state_wrapper = None
            
            # Record timestep
            self.recorders[i].record(
                state=state_wrapper,
                action=action,
                reward=reward,
                info=info,
                done=done,
            )
            
            self.total_steps += 1
            
            # Handle episode completion
            if done:
                self.recorders[i].end_episode()
                self.recorders[i].start_episode()
                self.episode_counts[i] += 1
                self.total_episodes += 1
                completed += 1
        
        return completed
    
    def end_episodes(self) -> None:
        """End recording for all environments.
        
        Only ends episodes that have actual recorded data to avoid
        adding empty episodes to the trajectory list.
        """
        for recorder in self.recorders:
            if recorder._recording:
                # Only end if the current episode has data
                # This prevents empty episodes from being saved
                if recorder.current_episode is not None and len(recorder.current_episode) > 0:
                    recorder.end_episode()
                else:
                    # Just clear the recording state without saving empty episode
                    recorder.current_episode = None
                    recorder._recording = False
    
    def get_all_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Get all recorded trajectories from all environments.
        
        Returns:
            List of trajectory dictionaries (excludes empty trajectories).
        """
        all_trajectories = []
        for recorder in self.recorders:
            for traj in recorder.get_trajectories():
                # Only include non-empty trajectories
                if traj and any(len(v) > 0 if hasattr(v, '__len__') else True for v in traj.values()):
                    all_trajectories.append(traj)
        return all_trajectories
    
    def save(
        self,
        path: Path,
        format: str = "pkl",
        compress: bool = True,
    ) -> None:
        """Save all recorded data to file.
        
        Args:
            path: Output file path.
            format: Output format ('npz', 'pkl', or 'hdf5').
            compress: Whether to compress the output.
        """
        import pickle
        
        path = Path(path)
        
        # Collect all trajectories
        trajectories = self.get_all_trajectories()
        
        data = {
            "trajectories": trajectories,
            "_metadata": {
                "num_episodes": self.total_episodes,
                "total_steps": self.total_steps,
                "num_envs": self.num_envs,
                "recording_components": self.recording_components,
                "format": "trajectories",
            }
        }
        
        if format == "pkl":
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        elif format == "npz":
            # For npz, we need to handle trajectories differently
            # Convert list of dicts to dict of lists
            if trajectories:
                all_data = {}
                for key in trajectories[0].keys():
                    try:
                        all_data[key] = np.concatenate([t[key] for t in trajectories if key in t])
                    except (ValueError, KeyError):
                        pass  # Skip keys that can't be concatenated
                
                # Add episode boundaries
                episode_lengths = [len(t.get('rewards', [])) for t in trajectories]
                all_data['episode_lengths'] = np.array(episode_lengths)
                
                if compress:
                    np.savez_compressed(path, **all_data)
                else:
                    np.savez(path, **all_data)
            else:
                np.savez(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pkl' or 'npz'.")
        
        print(f"Saved {self.total_episodes} episodes ({self.total_steps} steps) to {path}")
    
    @property
    def num_episodes(self) -> int:
        """Total number of completed episodes."""
        return self.total_episodes
    
    def clear(self) -> None:
        """Clear all recorded data."""
        for recorder in self.recorders:
            recorder.clear()
        self.episode_counts.fill(0)
        self.total_episodes = 0
        self.total_steps = 0


class StateSnapshotWrapper:
    """Wrapper to make StateSnapshot compatible with RolloutRecorder.
    
    This class provides the interface that RolloutRecorder expects,
    mapping StateSnapshot attributes to the expected format.
    """
    
    def __init__(
        self, 
        snapshot: Any, 
        custom_extractors: Dict[str, Callable] = None,
        env_obs: Optional[np.ndarray] = None,
    ):
        """Initialize wrapper.
        
        Args:
            snapshot: StateSnapshot object from vec_env.
            custom_extractors: Custom data extraction functions.
            env_obs: Environment observation (what the policy sees).
        """
        self._snapshot = snapshot
        self._custom_extractors = custom_extractors or {}
        self._env_obs = env_obs
        
        # Create a simple object that behaves like RawState
        self.raw = self._create_raw_state()
        self.derived = self._create_derived_state()
        self.accurate = self._create_accurate_state()
        
        # Expose default_dof_pos for RolloutRecorder._get_default_dof_pos()
        self.default_dof_pos = snapshot.default_dof_pos
        
        # Create a cfg-like object for compatibility with RolloutRecorder
        self.cfg = self._create_cfg()
    
    def _create_raw_state(self):
        """Create raw state-like object."""
        class RawStateLike:
            pass
        
        raw = RawStateLike()
        raw.pos_world = self._snapshot.pos_world
        raw.quat = self._snapshot.quat
        raw.vel_world = self._snapshot.vel_world
        raw.vel_body = self._snapshot.vel_body
        raw.ang_vel_world = self._snapshot.ang_vel_world
        raw.ang_vel_body = self._snapshot.ang_vel_body
        raw.dof_pos = self._snapshot.dof_pos
        raw.dof_vel = self._snapshot.dof_vel
        raw.gyros = self._snapshot.gyros
        raw.accs = self._snapshot.accs
        raw.quats = self._snapshot.quats
        raw.contact_floor_balls = self._snapshot.contact_floor_balls
        raw.contact_floor_geoms = self._snapshot.contact_floor_geoms
        raw.contact_floor_socks = self._snapshot.contact_floor_socks
        return raw
    
    def _create_derived_state(self):
        """Create derived state-like object."""
        class DerivedStateLike:
            pass
        
        derived = DerivedStateLike()
        derived.projected_gravity = self._snapshot.projected_gravity
        derived.projected_gravities = self._snapshot.projected_gravities
        derived.height = self._snapshot.height
        derived.heading = self._snapshot.heading
        derived.speed = self._snapshot.speed
        return derived
    
    def _create_accurate_state(self):
        """Create accurate state-like object."""
        class AccurateStateLike:
            pass
        
        accurate = AccurateStateLike()
        accurate.vel_world = self._snapshot.accurate_vel_world
        accurate.pos_world = self._snapshot.accurate_pos_world
        accurate.vel_body = self._snapshot.accurate_vel_body
        accurate.ang_vel_body = self._snapshot.accurate_ang_vel_body
        return accurate
    
    def _create_cfg(self):
        """Create cfg-like object for compatibility with RolloutRecorder.
        
        The RolloutRecorder._get_default_dof_pos() method checks for
        state.cfg.control.default_dof_pos, so we create a minimal
        cfg object that provides this attribute.
        """
        class ControlLike:
            pass
        
        class CfgLike:
            pass
        
        control = ControlLike()
        control.default_dof_pos = self._snapshot.default_dof_pos
        
        cfg = CfgLike()
        cfg.control = control
        
        return cfg
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to snapshot."""
        if name.startswith('_'):
            raise AttributeError(name)
        
        # Check raw state
        if hasattr(self.raw, name):
            return getattr(self.raw, name)
        
        # Check derived state
        if hasattr(self.derived, name):
            return getattr(self.derived, name)
        
        # Check accurate state (with prefix handling)
        if name.startswith('accurate_'):
            attr_name = name.replace('accurate_', '')
            if hasattr(self.accurate, attr_name):
                return getattr(self.accurate, attr_name)
        
        if hasattr(self.accurate, name):
            return getattr(self.accurate, name)
        
        # Check snapshot directly
        if hasattr(self._snapshot, name):
            return getattr(self._snapshot, name)
        
        # Return None for unknown attributes (don't raise error)
        return None
    
    def get_observation(self, insert: bool = False) -> Optional[np.ndarray]:
        """Get the environment observation.
        
        Args:
            insert: Ignored (for compatibility with State interface).
            
        Returns:
            The environment observation if available, None otherwise.
        """
        if self._env_obs is not None:
            return self._env_obs.copy()
        return None


def main():
    """Main function to record robot rollouts in parallel."""
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Import dependencies
    global CrossQ, ConfigRegistry, MetaMachine, RayVecMetaMachine, StateSnapshot
    
    try:
        from capyrl import CrossQ
    except ImportError:
        print("Error: CapyRL is required for policy inference.")
        print("Install with: pip install git+https://github.com/Chenaah/CapyRL.git")
        sys.exit(1)
    
    try:
        import ray
    except ImportError:
        print("Error: Ray is required for parallel environments.")
        print("Install with: pip install ray")
        sys.exit(1)
    
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.vec_env import RayVecMetaMachine, StateSnapshot
    
    # Resolve model path
    try:
        if POLICY_PATH:
            model_path = Path(POLICY_PATH)
            if not model_path.exists():
                print(f"Error: Policy file not found: {POLICY_PATH}")
                return
            print(f"Using local policy file: {model_path}")
        else:
            model_path = checkpoint_manager.get_checkpoint(MODEL)
    except Exception as e:
        print(f"Error resolving model: {e}")
        return
    
    # Load the trained policy
    print(f"\nLoading policy from: {model_path}")
    try:
        model = CrossQ.load_pkl(str(model_path), env=None, device=DEVICE)
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return
    
    # Create vectorized environment
    print(f"\nCreating {NUM_ENVS} parallel environments with config: {CONFIG}")
    cfg = ConfigRegistry.create_from_name(CONFIG)
    cfg.simulation.video_record_interval = None
    
    vec_env = RayVecMetaMachine(
        cfg,
        num_envs=NUM_ENVS,
        device=DEVICE,
        num_cpus_per_env=NUM_CPUS_PER_ENV,
        num_gpus_per_env=NUM_GPUS_PER_ENV,
        ray_temp_dir=RAY_TEMP_DIR,
        use_torch=False,  # Use numpy for easier handling
    )
    
    print(f"Environment info: num_obs={vec_env.num_obs}, num_actions={vec_env.num_actions}, num_modules={vec_env.num_modules}")
    
    # Create custom extractors based on actual number of modules
    custom_extractors = create_custom_extractors(vec_env.num_modules)
    
    # Create batch rollout recorder
    recorder = BatchRolloutRecorder(
        num_envs=NUM_ENVS,
        recording_components=RECORDING_COMPONENTS,
        separate_components=True,
        include_actions=True,
        include_rewards=True,
        include_infos=False,
        include_env_obs=True,
        custom_extractors=custom_extractors,
        action_as_actuator_command=True,
    )
    
    print(f"\nRecording components: {recorder.recording_components}")
    print(f"Custom extractors: {list(custom_extractors.keys())}")
    print(f"Target: {NUM_EPISODES} episodes")
    
    # Initialize environments with state snapshots
    print(f"\nInitializing {NUM_ENVS} environments...")
    obs, _, initial_states = vec_env.reset_with_states(seed=SEED)
    recorder.start_episodes()
    
    # Record episodes
    print(f"\nRecording rollouts in parallel with full state information...")
    start_time = time.time()
    step_count = 0
    
    rewards_accumulated = np.zeros(NUM_ENVS)
    
    try:
        while recorder.num_episodes < NUM_EPISODES:
            # Policy inference (batch)
            actions = model.predict(obs)
            
            # Step all environments and get state snapshots
            next_obs, _, rewards, dones, infos, states = vec_env.step_with_states(actions)
            
            # Record batch with actual state snapshots
            completed = recorder.record_batch(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                infos=infos.get('original_infos', None),
                env_obs=obs,
            )
            
            # Update for next step
            obs = next_obs
            step_count += NUM_ENVS
            rewards_accumulated += rewards
            
            # Reset accumulated rewards for completed episodes
            dones_np = np.array(dones, dtype=bool)
            rewards_accumulated[dones_np] = 0
            
            # Print progress
            if VERBOSE and (completed > 0 or step_count % (100 * NUM_ENVS) == 0):
                elapsed = time.time() - start_time
                eps_per_sec = recorder.num_episodes / elapsed if elapsed > 0 else 0
                steps_per_sec = step_count / elapsed if elapsed > 0 else 0
                print(
                    f"Episodes: {recorder.num_episodes}/{NUM_EPISODES} | "
                    f"Steps: {step_count} | "
                    f"Time: {elapsed:.1f}s | "
                    f"Eps/s: {eps_per_sec:.2f} | "
                    f"Steps/s: {steps_per_sec:.0f}"
                )
    
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    
    finally:
        # End any in-progress episodes
        recorder.end_episodes()
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    
    # Save recorded data
    output_path = Path(OUTPUT)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{OUTPUT_FORMAT}")
    
    print(f"\nSaving recordings to: {output_path}")
    recorder.save(output_path, format=OUTPUT_FORMAT)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Recording Summary")
    print("=" * 60)
    print(f"Parallel environments: {NUM_ENVS}")
    print(f"Episodes recorded: {recorder.num_episodes}")
    print(f"Total steps: {recorder.total_steps}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Episodes per second: {recorder.num_episodes / elapsed_time:.2f}")
    print(f"Steps per second: {recorder.total_steps / elapsed_time:.0f}")
    print(f"Output file: {output_path}")
    
    # Show example of loading data
    print("\n" + "=" * 60)
    print("Loading Example")
    print("=" * 60)
    data = RolloutRecorder.load(output_path)
    
    if "trajectories" in data:
        trajectories = data["trajectories"]
        print(f"Loaded {len(trajectories)} trajectories")
        
        if trajectories:
            example = trajectories[0]
            print(f"Example trajectory keys: {list(example.keys())}")
            
            # Show some statistics
            total_reward = sum(t['rewards'].sum() for t in trajectories if 'rewards' in t)
            avg_reward = total_reward / len(trajectories) if trajectories else 0
            print(f"Average episode reward: {avg_reward:.3f}")
            
            avg_length = sum(len(t.get('rewards', [])) for t in trajectories) / len(trajectories) if trajectories else 0
            print(f"Average episode length: {avg_length:.1f}")
            
            # Show sample state data
            if 'accurate_vel_world' in example:
                print(f"accurate_vel_world shape: {example['accurate_vel_world'].shape}")
            
            # Show custom extractor data
            for key in custom_extractors.keys():
                if key in example:
                    print(f"{key} shape: {example[key].shape}")
    
    # Cleanup
    vec_env.close()
    print("\nRecording completed successfully!")


if __name__ == "__main__":
    main()
