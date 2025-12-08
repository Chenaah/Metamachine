

"""
Policy Inference Example

This example demonstrates how to load a trained policy and run inference
in the MetaMachine simulation environment. It requires the CapyRL library
for loading CrossQ policies.

The script uses MetaMachine's checkpoint manager to handle model downloads:
1. Use a model name from the registry (auto-downloads if needed)
2. Provide a local path to a policy file
3. Provide a direct URL to download from

Usage:
    # Run with default settings (uses three_modules_run_policy)
    python policy_inference.py
    
    # List available models
    python policy_inference.py --list_models
    
    # Use a different model from registry
    python policy_inference.py --model another_model
    
    # Use a local policy file
    python policy_inference.py --policy_path ./trained_policy.pkl
    
    # Download from a custom URL
    python policy_inference.py --model_url https://example.com/policy.pkl
    
    # Customize simulation settings
    python policy_inference.py --num_steps 500 --render_mode mp4 --seed 123
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from metamachine.utils.checkpoint_manager import CheckpointManager

# Lazy import of optional dependencies
CrossQ = None
ConfigRegistry = None
MetaMachine = None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run policy inference in MetaMachine environment"
    )
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default="three_modules_run_policy",
        help="Name of registered model to load (default: three_modules_run_policy)",
    )
    model_group.add_argument(
        "--policy_path",
        type=str,
        help="Path to local policy file (.pkl)",
    )
    model_group.add_argument(
        "--model_url",
        type=str,
        help="Direct URL to download model from",
    )
    
    # Checkpoint manager options
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List all registered models and exit",
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="Show checkpoint cache information and exit",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of model even if cached",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="example_three_modules",
        help="Environment configuration name (default: example_three_modules)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Number of simulation steps to run (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="viewer",
        choices=["viewer", "mp4", "none"],
        help="Rendering mode (default: viewer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run policy on (default: cpu)",
    )
    parser.add_argument(
        "--print_interval",
        type=int,
        default=20,
        help="Print progress every N steps (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print observation values at each step",
    )
    return parser.parse_args()


def main():
    """Main function to run policy inference."""
    args = parse_args()
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Handle info commands (don't require CapyRL)
    if args.list_models:
        checkpoint_manager.print_models()
        return
    
    if args.cache_info:
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
        if args.policy_path:
            # Local file
            model_path = Path(args.policy_path)
            if not model_path.exists():
                print(f"Error: Policy file not found: {args.policy_path}")
                return
            print(f"Using local policy file: {model_path}")
        elif args.model_url:
            # Direct URL
            model_path = checkpoint_manager.download_from_url(
                args.model_url,
                force_download=args.force_download,
            )
        else:
            # Registered model (default or specified)
            model_path = checkpoint_manager.get_checkpoint(
                args.model,
                force_download=args.force_download,
            )
    except Exception as e:
        print(f"Error resolving model: {e}")
        return

    # Load the trained policy
    print(f"\nLoading policy from: {model_path}")
    try:
        model = CrossQ.load_pkl(str(model_path), env=None, device=args.device)
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Create environment configuration
    print(f"Creating environment with config: {args.config}")
    cfg = ConfigRegistry.create_from_name(args.config)

    # Initialize the MetaMachine simulation environment
    env = MetaMachine(cfg)
    env.render_mode = args.render_mode

    # Reset environment to initial state
    print(f"Resetting environment with seed: {args.seed}")
    obs, _ = env.reset(seed=args.seed)

    # Main control loop
    print(f"Starting inference for {args.num_steps} steps...")
    for step in range(args.num_steps):
        t0 = time.time()

        # Get action from policy
        action = model.predict(obs.reshape(1, -1))

        # Step the environment forward one timestep
        obs, reward, done, truncated, info = env.step(action[0])

        # Print observation if verbose mode is enabled
        if args.verbose:
            print(f"Step {step} - Observation: {obs}")

        # Render the current state
        env.render()

        # Print progress at specified intervals
        if step % args.print_interval == 0:
            print(
                f"Step {step}/{args.num_steps}: "
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


