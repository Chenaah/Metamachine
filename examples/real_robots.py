#!/usr/bin/env python3
"""
Real Robot Control Example

This script demonstrates how to control real robots using the RealMetaMachine
environment with capybarish for ESP32 communication.

Features:
- Loads config with expected module_ids for ordered action mapping
- Auto-discovers and validates ESP32 modules (both active and sensor-only)
- Supports passive sensor modules (e.g., dedicated distance sensors)
- Configurable global state sources (main IMU, goal distance)
- Optional Rich dashboard for real-time monitoring
- Keyboard controls: e=enable, d=disable, r=restart, c=calibrate, q=quit

Usage:
    # Basic usage with default config
    python examples/real_robots.py
    
    # With custom config
    python examples/real_robots.py --config path/to/config.yaml
    
    # With sinusoidal test motion
    python examples/real_robots.py --test-motion
    
    # Load and run a trained policy
    python examples/real_robots.py --policy path/to/policy.pt

Before running:
    1. Configure ESP32 modules with their module_ids (0, 1, 2, ...)
    2. Configure ESP32 to send data to this computer's IP on port 6666
    3. Ensure the module_ids in config match your physical modules

Module Types:
    - Active modules (module_ids): Receive motor commands, action[i] -> module_ids[i]
    - Sensor modules (sensor_module_ids): No motor commands, sensor data only
    
    Example config for 3 active modules + 1 sensor module:
        real:
          module_ids: [0, 1, 2]       # 3 active modules
          sensor_module_ids: [100]    # 1 dedicated distance sensor
          sources:
            main_imu: 0               # Module 0 provides main quat/gyro
            goal_distance: 100        # Module 100 provides global goal_distance

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Default config path
DEFAULT_CONFIG = str(
    PROJECT_ROOT / "metamachine" / "environments" / "configs" / 
    "default_configs" / "real_three_modules.yaml"
)


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[Signal] Received interrupt. Shutting down...")
    sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real Robot Control with RealMetaMachine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python real_robots.py
    
    # Run with custom config
    python real_robots.py --config my_robot.yaml
    
    # Run sinusoidal test motion
    python real_robots.py --test-motion --amplitude 0.5 --frequency 0.3
    
    # Run with trained policy
    python real_robots.py --policy logs/experiment/policy.pt
    
Keyboard Controls (during operation):
    e - Enable motors
    d - Disable motors
    r - Restart motors
    c - Calibrate motors
    q - Quit
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to robot configuration YAML file"
    )
    
    parser.add_argument(
        "--test-motion",
        action="store_true",
        help="Run sinusoidal test motion (no policy required)"
    )
    
    parser.add_argument(
        "--amplitude", "-a",
        type=float,
        default=0.5,
        help="Sinusoidal motion amplitude (radians)"
    )
    
    parser.add_argument(
        "--frequency", "-f",
        type=float,
        default=0.3,
        help="Sinusoidal motion frequency (Hz)"
    )
    
    parser.add_argument(
        "--policy", "-p",
        type=str,
        default=None,
        help="Path to trained policy checkpoint"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        default=None,
        help="Path to training log directory (loads config and checkpoint from there)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load from log-dir: 'latest', 'final', 'best', or step number"
    )
    
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable Rich dashboard"
    )
    
    parser.add_argument(
        "--duration", "-t",
        type=float,
        default=None,
        help="Run duration in seconds (None = run until quit)"
    )
    
    return parser.parse_args()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    cfg = ConfigRegistry.create_from_file(config_path)
    return cfg


def create_environment(cfg):
    """Create the real robot environment."""
    from metamachine.environments.env_real import RealMetaMachine
    
    # Ensure mode is set to real
    if cfg.environment.get("mode", "sim") != "real":
        print("[Warning] Config mode is not 'real'. Forcing mode=real.")
        cfg.environment.mode = "real"
    
    env = RealMetaMachine(cfg)
    return env


def load_policy(policy_path: str, env):
    """Load a trained policy checkpoint."""
    import torch
    
    if not os.path.exists(policy_path):
        print(f"Error: Policy file not found: {policy_path}")
        return None
    
    print(f"Loading policy from: {policy_path}")
    
    try:
        # Try loading as a CrossQ/JAX policy
        checkpoint = torch.load(policy_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "actor" in checkpoint:
            actor = checkpoint["actor"]
        elif "policy" in checkpoint:
            actor = checkpoint["policy"]
        else:
            print("[Warning] Unknown checkpoint format. Keys:", list(checkpoint.keys()))
            return None
        
        print("Policy loaded successfully!")
        return actor
        
    except Exception as e:
        print(f"Error loading policy: {e}")
        return None


def run_sinusoidal_test(env, amplitude: float, frequency: float, duration: float = None):
    """Run sinusoidal test motion.
    
    Args:
        env: RealMetaMachine environment
        amplitude: Motion amplitude in radians
        frequency: Motion frequency in Hz
        duration: Duration in seconds (None = run until interrupt)
    """
    print("\n" + "=" * 60)
    print("Sinusoidal Test Motion")
    print("=" * 60)
    print(f"  Amplitude: {amplitude} rad")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Generate sinusoidal action
            # Each motor gets the same sine wave (can be customized)
            num_actions = env.action_space.shape[0]
            phase = 2 * np.pi * frequency * elapsed
            
            # Create action with phase offset per motor for gait-like motion
            action = np.zeros(num_actions)
            for i in range(num_actions):
                phase_offset = (2 * np.pi * i) / num_actions  # Distribute phases
                action[i] = amplitude * np.sin(phase + phase_offset)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Reward: {reward:.4f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                print(f"\n[Episode ended] Done={done}, Truncated={truncated}")
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nTest completed: {step_count} steps in {elapsed:.1f}s")


def run_policy(env, policy, duration: float = None):
    """Run a trained policy (PyTorch model).
    
    Args:
        env: RealMetaMachine environment
        policy: Loaded policy model (PyTorch)
        duration: Duration in seconds (None = run until interrupt)
    """
    import torch
    
    print("\n" + "=" * 60)
    print("Running Trained Policy")
    print("=" * 60)
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor)
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                action = action.squeeze()
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Episode Reward: {episode_reward:.2f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                print(f"\n[Episode ended] Reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nPolicy run completed: {step_count} steps in {elapsed:.1f}s")


def run_sb3_policy(env, model, duration: float = None, deterministic: bool = True):
    """Run a trained SB3 policy.
    
    Args:
        env: RealMetaMachine environment
        model: Loaded SB3 model (CrossQ, SAC, PPO, etc.)
        duration: Duration in seconds (None = run until interrupt)
        deterministic: Use deterministic actions (no exploration noise)
    """
    print("\n" + "=" * 60)
    print("Running SB3 Policy")
    print("=" * 60)
    print(f"  Duration: {'infinite' if duration is None else f'{duration}s'}")
    print(f"  Deterministic: {deterministic}")
    print("=" * 60)
    print("\nPress 'e' to enable motors, 'q' to quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    episode_reward = 0
    episode_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Get action from SB3 model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # Print status periodically
            if step_count % 100 == 0:
                print(f"\r[Step {step_count}] Time: {elapsed:.1f}s, "
                      f"Episode Reward: {episode_reward:.2f}", end="", flush=True)
            
            # Check for episode end
            if done or truncated:
                episode_count += 1
                print(f"\n[Episode {episode_count}] Reward: {episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n\nPolicy run completed: {step_count} steps, "
              f"{episode_count} episodes in {elapsed:.1f}s")


def run_idle(env, duration: float = None):
    """Run in idle mode - just monitor modules without motion.
    
    Useful for testing connectivity and calibration.
    """
    print("\n" + "=" * 60)
    print("Idle Mode - Monitoring Only")
    print("=" * 60)
    print("  Motors will not move automatically")
    print("  Use this mode for connectivity testing and calibration")
    print("=" * 60)
    print("\nControls:")
    print("  e - Enable motors")
    print("  d - Disable motors")
    print("  r - Restart motors")
    print("  c - Calibrate motors")
    print("  s - Print status")
    print("  q - Quit")
    
    # Reset environment
    obs, info = env.reset()
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            if duration is not None and elapsed >= duration:
                print(f"\n[Done] Reached duration limit ({duration}s)")
                break
            
            # Send zero action to maintain communication
            num_actions = env.action_space.shape[0]
            action = np.zeros(num_actions)
            
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Handle special keyboard input for status
            if hasattr(env, 'input_key') and env.input_key == 's':
                env.print_status()
                env.input_key = ""
            
            if done or truncated:
                obs, info = env.reset()
    
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\nIdle mode ended: {step_count} steps in {elapsed:.1f}s")


def main():
    """Main entry point."""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_args()
    
    print("=" * 60)
    print("Real Robot Control - RealMetaMachine")
    print("=" * 60)
    
    # Check if loading from training log directory
    if args.log_dir:
        # Use load_from_checkpoint utility for seamless loading
        print(f"\nLoading from training log: {args.log_dir}")
        try:
            from metamachine.utils.sb3_utils import load_from_checkpoint
            
            env, model, cfg = load_from_checkpoint(
                args.log_dir,
                checkpoint=args.checkpoint,
                real_robot=True,
            )
            
            # Override dashboard setting if requested
            if args.no_dashboard:
                if hasattr(cfg, 'real') and cfg.real:
                    cfg.real.enable_dashboard = False
            
            try:
                if model is not None:
                    run_sb3_policy(env, model, duration=args.duration)
                else:
                    print("No model found in log directory. Running idle mode.")
                    run_idle(env, duration=args.duration)
            finally:
                print("\nCleaning up...")
                env.close()
            return
            
        except ImportError as e:
            print(f"Error: Could not import sb3_utils: {e}")
            print("Falling back to manual config loading...")
    
    # Load configuration from file
    print(f"\nLoading config: {args.config}")
    cfg = load_config(args.config)
    
    # Override dashboard setting if requested
    if args.no_dashboard:
        if "real" not in cfg:
            cfg.real = {}
        cfg.real.enable_dashboard = False
    
    # Create environment
    print("\nCreating RealMetaMachine environment...")
    env = create_environment(cfg)
    
    try:
        if args.test_motion:
            # Run sinusoidal test motion
            run_sinusoidal_test(
                env,
                amplitude=args.amplitude,
                frequency=args.frequency,
                duration=args.duration
            )
        
        elif args.policy:
            # Load and run policy
            policy = load_policy(args.policy, env)
            if policy is not None:
                run_policy(env, policy, duration=args.duration)
            else:
                print("Failed to load policy. Running idle mode instead.")
                run_idle(env, duration=args.duration)
        
        else:
            # Run idle mode (monitoring only)
            run_idle(env, duration=args.duration)
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        env.close()


if __name__ == "__main__":
    main()
