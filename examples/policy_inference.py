

"""
Policy Inference Example

This example demonstrates how to load a trained policy and run inference
in the MetaMachine simulation environment. It requires the CapyRL library
for loading CrossQ policies.

The script uses MetaMachine's checkpoint manager to handle model downloads:
1. Use a model name from the registry (auto-downloads if needed)
2. Provide a local path to a policy file
3. Provide a direct URL to download from

Configuration:
    Modify the global variables at the top of this file to customize behavior:
    
    # To use a different model from registry:
    MODEL = "your_model_name"
    
    # To use a local policy file:
    POLICY_PATH = "./path/to/your/policy.pkl"
    MODEL = None  # Set this to None when using POLICY_PATH
    
    # To download from a custom URL:
    MODEL_URL = "https://example.com/policy.pkl"
    MODEL = None  # Set this to None when using MODEL_URL
    
    # To customize simulation settings:
    NUM_STEPS = 500
    RENDER_MODE = "mp4"  # or "viewer" or "none"
    SEED = 123
    
    # To list available models or show cache info:
    LIST_MODELS = True  # or CACHE_INFO = True

Usage:
    # Simply run the script after configuring the global variables
    python policy_inference.py
"""

import sys
import time
from pathlib import Path

import numpy as np

from metamachine.utils.checkpoint_manager import CheckpointManager

# Configuration - modify these variables to change behavior
# Model loading options (use only one at a time)
MODEL = "three_modules_run_policy"  # Name of registered model to load
POLICY_PATH = None  # Path to local policy file (.pkl)
MODEL_URL = None  # Direct URL to download model from

# Checkpoint manager options
LIST_MODELS = False  # List all registered models and exit
CACHE_INFO = False  # Show checkpoint cache information and exit
FORCE_DOWNLOAD = False  # Force re-download of model even if cached

# Environment and simulation settings
CONFIG = "example_three_modules"  # Environment configuration name
NUM_STEPS = 1000  # Number of simulation steps to run
SEED = 42  # Random seed for reproducibility
RENDER_MODE = "viewer"  # Rendering mode: "viewer", "mp4", or "none"
DEVICE = "cpu"  # Device to run policy on: "cpu" or "cuda"
PRINT_INTERVAL = 20  # Print progress every N steps
VERBOSE = False  # Print observation values at each step

# Lazy import of optional dependencies
CrossQ = None
ConfigRegistry = None
MetaMachine = None



def main():
    """Main function to run policy inference."""
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Handle info commands (don't require CapyRL)
    if LIST_MODELS:
        checkpoint_manager.print_models()
        return
    
    if CACHE_INFO:
        checkpoint_manager.print_cache_info()
        return
    
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
            # Local file
            model_path = Path(POLICY_PATH)
            if not model_path.exists():
                print(f"Error: Policy file not found: {POLICY_PATH}")
                return
            print(f"Using local policy file: {model_path}")
        elif MODEL_URL:
            # Direct URL
            model_path = checkpoint_manager.download_from_url(
                MODEL_URL,
                force_download=FORCE_DOWNLOAD,
            )
        else:
            # Registered model (default or specified)
            model_path = checkpoint_manager.get_checkpoint(
                MODEL,
                force_download=FORCE_DOWNLOAD,
            )
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

    # Create environment configuration
    print(f"Creating environment with config: {CONFIG}")
    cfg = ConfigRegistry.create_from_name(CONFIG)

    # Initialize the MetaMachine simulation environment
    env = MetaMachine(cfg)
    env.render_mode = RENDER_MODE

    # Reset environment to initial state
    print(f"Resetting environment with seed: {SEED}")
    obs, _ = env.reset(seed=SEED)

    # Main control loop
    print(f"Starting inference for {NUM_STEPS} steps...")
    for step in range(NUM_STEPS):
        t0 = time.time()

        # Get action from policy
        action = model.predict(obs.reshape(1, -1))

        # Step the environment forward one timestep
        obs, reward, done, truncated, info = env.step(action[0])

        # Print observation if verbose mode is enabled
        if VERBOSE:
            print(f"Step {step} - Observation: {obs}")

        # Render the current state
        env.render()

        # Print progress at specified intervals
        if step % PRINT_INTERVAL == 0:
            print(
                f"Step {step}/{NUM_STEPS}: "
                f"reward={reward:.3f}, done={done}, truncated={truncated}"
            )

        # Maintain real-time simulation speed
        elapsed = time.time() - t0
        sleep_time = max(0, env.cfg.control.dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Check for episode termination
        if done or truncated:
            print(f"Episode ended at step {step}")
            print(f"Final reward: {reward:.3f}")
            break

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()


