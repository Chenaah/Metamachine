"""
Rollout Recording Example

This example demonstrates how to record robot rollouts while using a trained
policy for inference. The key feature is the ability to record different
observation data than what the policy sees.

Use cases:
1. Record privileged state information (ground truth positions, velocities)
   while the policy uses estimated/processed observations
2. Record additional data like contact forces, energy consumption
3. Save rollouts for analysis, visualization, or imitation learning

The script uses MetaMachine's RolloutRecorder to capture:
- Full state information (position, velocity, joint states)
- Actions taken by the policy
- Rewards received
- Custom data via extractors

Configuration:
    Modify the global variables at the top of this file to customize behavior:
    
    # To use a different model from registry:
    MODEL = "your_model_name"
    
    # To use a local policy file:
    POLICY_PATH = "./path/to/your/policy.pkl"
    MODEL = None  # Set this to None when using POLICY_PATH
    
    # To record multiple episodes:
    NUM_EPISODES = 10
    
    # To save in different format:
    OUTPUT_FORMAT = "hdf5"
    OUTPUT = "rollouts.h5"
    
    # To enable rendering and verbose output:
    RENDER = True
    VERBOSE = True
    
    # To record specific components:
    RECORDING_COMPONENTS = ["gyros", "projected_gravities"]

Usage:
    # Simply run the script after configuring the global variables
    python record_rollout.py
"""

import sys
import time
from pathlib import Path

import numpy as np

from metamachine.utils.checkpoint_manager import CheckpointManager
from metamachine.utils.rollout_recorder import RolloutRecorder, StateSnapshot

# Configuration - modify these variables to change behavior
# Model loading options
MODEL = "three_modules_run_policy"  # Name of registered model to load
POLICY_PATH = None  # Path to local policy file (.pkl)

# Environment options  
CONFIG = "example_three_modules"  # Environment configuration name

# Recording options
NUM_EPISODES = 1000  # Number of episodes to record
MAX_STEPS = 1000  # Maximum steps per episode
OUTPUT = "rollouts.npz"  # Output file path
OUTPUT_FORMAT = "npz"  # Output format: "npz", "pkl", or "hdf5"
RECORDING_COMPONENTS = ["accurate_vel_world"]  # State components to record (None = all privileged components)

# Simulation options
SEED = 42  # Random seed for reproducibility
RENDER = False  # Render the simulation (slower but visual)
VERBOSE = False  # Print detailed information during recording

# Lazy import of optional dependencies
CrossQ = None
ConfigRegistry = None
MetaMachine = None




def create_custom_extractors():
    """Create custom data extractors for recording additional information.
    
    Returns:
        Dictionary of extractor functions.
    """
    return {
        
        # Check if robot is flying (no ground contact)
        f"module{i}": lambda s: np.concatenate([s.projected_gravities[i],
                                                s.gyros[i],
                                                s.dof_pos[i:i+1],
                                                s.dof_vel[i:i+1]])
        for i in range(3)
    }


def main():
    """Main function to record robot rollouts."""
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Import dependencies needed for inference
    global CrossQ, ConfigRegistry, MetaMachine
    try:
        from capyrl import CrossQ
    except ImportError:
        print("Error: CapyRL is required for policy inference.")
        print("Install with: pip install git+https://github.com/Chenaah/CapyRL.git")
        sys.exit(1)
    
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    
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
        model = CrossQ.load_pkl(str(model_path), env=None, device="cpu")
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Create environment
    print(f"\nCreating environment with config: {CONFIG}")
    cfg = ConfigRegistry.create_from_name(CONFIG)
    env = MetaMachine(cfg)
    env.render_mode = "viewer" if RENDER else "none"

    # Create rollout recorder
    # You can specify which components to record
    # If None, it uses the default privileged components
    recorder = RolloutRecorder(
        recording_components=RECORDING_COMPONENTS,
        separate_components=True,
        include_actions=True,
        include_rewards=True,
        include_infos=False,  # Set to True if you want full info dicts
        include_env_obs=True,  # Set to True if you want to record environment observations
        custom_extractors=create_custom_extractors(),
        action_as_actuator_command=True
    )
    
    print(f"\nRecording components: {recorder.recording_components}")
    print(f"Custom extractors: {list(recorder.custom_extractors.keys())}")

    # Record episodes
    print(f"\nRecording {NUM_EPISODES} episodes...")
    
    for episode in range(NUM_EPISODES):
        print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")
        
        # Reset environment
        obs, _ = env.reset(seed=SEED + episode)
        recorder.start_episode()
        
        total_reward = 0
        step = 0
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        for step in range(MAX_STEPS):
            t0 = time.time()
            
            # Policy uses the environment's observation
            action = model.predict(obs.reshape(1, -1))

            # Record using the state object (which has full privileged state)
            # The policy sees 'obs', but we record from env.state
            recorder.record(
                state=env.state,
                action=action[0],
                reward=reward,
                info=info,
                done=done or truncated,
            )
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action[0])
            
            total_reward += reward
            
            # Render if requested
            if RENDER:
                env.render()
                elapsed = time.time() - t0
                sleep_time = max(0, env.cfg.control.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Print progress
            if VERBOSE and step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
            
            if done or truncated:
                break
        
        recorder.end_episode()
        print(f"Episode {episode + 1}: {step + 1} steps, total reward: {total_reward:.3f}")
    
    # Save recorded data
    output_path = Path(OUTPUT)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{OUTPUT_FORMAT}")
    
    print(f"\nSaving recordings to: {output_path}")
    recorder.save(output_path, format=OUTPUT_FORMAT, as_trajectories=True)
    
    # Print summary
    print("\n=== Recording Summary ===")
    print(f"Episodes recorded: {recorder.num_episodes}")
    print(f"Total steps: {recorder.total_steps}")
    print(f"Output file: {output_path}")
    
    # Show example of loading data
    print("\n=== Loading Example ===")
    data = RolloutRecorder.load(output_path)
    print(f"Loaded data keys: {list(data.keys())}")
    if "observations" in data:
        print(f"Observations shape: {data['observations'].shape}")
    if "actions" in data:
        print(f"Actions shape: {data['actions'].shape}")
    if "rewards" in data:
        print(f"Total reward in recording: {np.sum(data['rewards']):.3f}")
    
    # Show custom data
    for key in recorder.custom_extractors.keys():
        if key in data:
            print(f"{key}: shape={data[key].shape}, mean={np.mean(data[key]):.4f}")

    env.close()
    print("\nRecording completed successfully!")


if __name__ == "__main__":
    main()
