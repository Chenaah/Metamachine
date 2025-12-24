#!/usr/bin/env python3
"""
Train a MetaMachine Robot with RSL-RL (PPO)

This script demonstrates how to train modular robots using:
1. RayVecMetaMachine for parallel environment execution
2. RSL-RL's OnPolicyRunner for PPO training
3. Integration with wandb for experiment tracking

The RayVecMetaMachine provides a vectorized environment that is compatible with
the rsl_rl VecEnv interface, enabling efficient parallel data collection.

Usage:
    # Train with default quadruped config
    python train_rsl_rl.py
    
    # Train with custom config file
    python train_rsl_rl.py --config /path/to/config.yaml
    
    # Train with more environments
    python train_rsl_rl.py --num-envs 4096 --device cuda:0
    
    # Resume from checkpoint
    python train_rsl_rl.py --resume --load-run "experiment-name" --checkpoint 1000

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Disable debug mode to reduce overhead
os.environ["METAMACHINE_DEBUG"] = "0"
os.environ["METAMACHINE_DEBUG_VERBOSE"] = "0"

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Default RSL-RL Configuration
# ============================================================================

def get_default_rsl_rl_config():
    """Get default RSL-RL training configuration.
    
    These parameters are tuned for locomotion tasks. You can override
    them via command line arguments or by modifying this function.
    
    Returns:
        dict: RSL-RL configuration dictionary.
    """
    return {
        # Runner settings
        "seed": 42,
        "run_name": "",
        "max_iterations": 10000,
        "save_interval": 500,
        "experiment_name": "metamachine",
        "num_steps_per_env": 24,  # Number of steps per env per update
        
        # Observation groups mapping
        # Maps observation groups from environment to observation sets used by the algorithm
        # - "policy": observations used by the actor network
        # - "critic": observations used by the critic network (can include privileged info)
        "obs_groups": {
            "policy": ["policy"],   # Actor uses "policy" observation group
            "critic": ["critic"],   # Critic uses "critic" observation group (same as policy for now)
        },
        
        # Policy settings (ActorCritic network)
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
        },
        
        # Algorithm settings (PPO)
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            # RND (Random Network Distillation) - disabled by default
            "rnd_cfg": None,
            # Symmetry augmentation - disabled by default
            "symmetry_cfg": None,
        },
        
        # Logging
        "logger": "tensorboard",  # tensorboard, neptune, wandb
    }


def get_load_path(root_dir: str, load_run: str = "-1", checkpoint: int = -1) -> str:
    """Get the path to load a checkpoint from.
    
    Args:
        root_dir: Root directory containing experiment logs.
        load_run: Name of the run to load. "-1" means latest.
        checkpoint: Checkpoint number to load. -1 means latest.
        
    Returns:
        Path to the checkpoint file.
    """
    import glob
    
    if load_run == "-1":
        # Find the latest run
        runs = sorted(glob.glob(os.path.join(root_dir, "*")))
        if not runs:
            raise ValueError(f"No runs found in {root_dir}")
        load_run = os.path.basename(runs[-1])
    
    run_dir = os.path.join(root_dir, load_run, "checkpoints")
    
    if checkpoint == -1:
        # Find the latest checkpoint
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")))
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_dir}")
        checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = os.path.join(run_dir, f"model_{checkpoint}.pt")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a MetaMachine robot using RSL-RL (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default quadruped config
    python train_rsl_rl.py
    
    # Train with custom config
    python train_rsl_rl.py --config basic_quadruped
    
    # Train with more environments and on GPU
    python train_rsl_rl.py --num-envs 4096 --device cuda:0
    
    # Resume training from checkpoint
    python train_rsl_rl.py --resume --load-run "metamachine-20231201-120000"
        """
    )
    
    # Environment arguments
    parser.add_argument(
        "--config",
        type=str,
        default="basic_quadruped",
        help="Config name or path to YAML file"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=5,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for training (cuda:0, cpu)"
    )
    parser.add_argument(
        "--num-cpus-per-env",
        type=float,
        default=0.1,
        help="CPU resources per environment"
    )
    parser.add_argument(
        "--num-gpus-per-env",
        type=float,
        default=0.0,
        help="GPU resources per environment"
    )
    
    # Training arguments
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum training iterations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix to append to experiment name"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--load-run",
        type=str,
        default=None,
        help="Run name to load checkpoint from"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=-1,
        help="Checkpoint number to load (-1 for latest)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/rsl_rl",
        help="Directory for training logs"
    )
    
    # Play-only mode
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="Only run inference with loaded policy, no training"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Import dependencies
    # =========================================================================
    print("=" * 60)
    print("MetaMachine RSL-RL Training")
    print("=" * 60)
    
    print("\n[1/4] Importing dependencies...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    
    try:
        import ray
        print(f"  Ray version: {ray.__version__}")
    except ImportError:
        print("ERROR: Ray is required. Install with: pip install ray")
        sys.exit(1)
    
    try:
        from rsl_rl.runners import OnPolicyRunner
        print("  RSL-RL: OK")
    except ImportError:
        print("ERROR: RSL-RL is required.")
        print("Install with: pip install rsl-rl")
        sys.exit(1)
    
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        if args.use_wandb:
            print("WARNING: wandb not available. Disabling wandb logging.")
            args.use_wandb = False
    
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.vec_env import RayVecMetaMachine
    
    print("  ✓ All dependencies loaded")
    
    # =========================================================================
    # Load configuration
    # =========================================================================
    print("\n[2/4] Loading environment configuration...")
    
    if os.path.exists(args.config):
        # Load from file
        cfg = ConfigRegistry.create_from_file(args.config)
        print(f"  Loaded from file: {args.config}")
    else:
        # Load from registry
        cfg = ConfigRegistry.create_from_name(args.config)
        print(f"  Loaded from registry: {args.config}")
    
    # Show video recording status from config (not overwritten)
    render_mode = cfg.simulation.get('render_mode', 'none')
    video_interval = cfg.simulation.get('video_record_interval', None)
    if render_mode == 'mp4' and video_interval:
        print(f"  Video recording: ENABLED (every {video_interval} episodes)")
    else:
        print(f"  Video recording: disabled (render_mode={render_mode})")
    
    print(f"  Robot type: {cfg.morphology.robot_type}")
    
    # Generate experiment name with naming pattern: {config_name}-{timestamp}
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.exp_name:
        exp_name = args.exp_name
    else:
        config_name = os.path.basename(args.config).replace(".yaml", "")
        exp_name = f"{config_name}-{timestamp}"
    
    if args.suffix:
        exp_name = f"{exp_name}-{args.suffix}"
    
    print(f"  Experiment name: {exp_name}")
    
    # Set experiment name in config so environment uses the same naming
    from omegaconf import OmegaConf
    if not hasattr(cfg, 'logging'):
        cfg.logging = OmegaConf.create({})
    cfg.logging.experiment_name = exp_name
    
    print("  ✓ Configuration loaded")
    
    # =========================================================================
    # Create vectorized environment
    # =========================================================================
    print("\n[3/4] Creating vectorized environment...")
    
    num_envs = 1 if args.play_only else args.num_envs
    print(f"  Number of environments: {num_envs}")
    print(f"  Device: {args.device}")
    
    vec_env = RayVecMetaMachine(
        cfg,
        num_envs=num_envs,
        device=args.device,
        num_cpus_per_env=args.num_cpus_per_env,
        num_gpus_per_env=args.num_gpus_per_env,
        use_torch=True,  # RSL-RL expects torch tensors
    )
    
    print(f"  Observation dim: {vec_env.num_obs}")
    print(f"  Action dim: {vec_env.num_actions}")
    print(f"  Max episode length: {vec_env.max_episode_length}")
    
    # Get log directory from primary environment
    # This is where videos and env artifacts are saved
    env_log_dir = vec_env.log_dir
    if env_log_dir:
        print(f"  Environment log directory: {env_log_dir}")
        # Use environment's log directory as the base for all logging
        log_dir = env_log_dir
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Check if video recording is enabled
        render_mode = cfg.simulation.get('render_mode', 'none')
        if render_mode == 'mp4':
            print(f"  ✓ Video recording enabled (render_mode: mp4)")
    else:
        # Fallback if no env log dir (e.g., render_mode: none)
        log_dir = os.path.abspath(os.path.join(args.log_dir, exp_name))
        os.makedirs(log_dir, exist_ok=True)
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"  Log directory: {log_dir}")
    
    # Save config to log directory
    config_path = os.path.join(ckpt_dir, "env_config.json")
    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"  Saved config to: {config_path}")
    except Exception as e:
        print(f"  Warning: Could not save config: {e}")
    
    print("  ✓ Environment created")
    
    # =========================================================================
    # Setup RSL-RL training
    # =========================================================================
    print("\n[4/4] Setting up RSL-RL training...")
    
    # Get training config
    train_cfg = get_default_rsl_rl_config()
    
    # Override with command line arguments
    train_cfg["seed"] = args.seed
    train_cfg["run_name"] = exp_name
    train_cfg["max_iterations"] = args.max_iterations
    train_cfg["resume"] = args.resume
    train_cfg["load_run"] = args.load_run if args.load_run else "-1"
    train_cfg["checkpoint"] = args.checkpoint
    
    # Save training config
    train_cfg_path = os.path.join(ckpt_dir, "train_config.json")
    with open(train_cfg_path, 'w') as f:
        json.dump(train_cfg, f, indent=2)
    print(f"  Saved training config to: {train_cfg_path}")
    
    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE and not args.play_only:
        print("  Initializing Weights & Biases...")
        wandb.tensorboard.patch(root_logdir=log_dir)
        wandb.init(project="metamachine-rsl-rl", name=exp_name)
        wandb.config.update(train_cfg)
        try:
            wandb.config.update({"env_config": config_dict})
        except:
            pass
    
    # Create the runner
    runner = OnPolicyRunner(
        vec_env,
        train_cfg,
        log_dir,
        device=args.device
    )
    
    # Load checkpoint if resuming
    if args.resume:
        resume_path = get_load_path(
            args.log_dir,
            load_run=train_cfg["load_run"],
            checkpoint=train_cfg["checkpoint"]
        )
        print(f"  Loading checkpoint from: {resume_path}")
        runner.load(resume_path)
    
    print("  ✓ Runner created")
    
    # =========================================================================
    # Train or play
    # =========================================================================
    if args.play_only:
        print("\n" + "=" * 60)
        print("Running inference (play-only mode)")
        print("=" * 60)
        
        policy = runner.get_inference_policy(device=args.device)
        
        obs = vec_env.reset(seed=args.seed)
        total_reward = 0.0
        episode_length = 0
        
        print("\nRunning episode...")
        for step in range(vec_env.max_episode_length):
            with torch.no_grad():
                actions = policy(obs)
                actions = torch.clamp(actions, -1.0, 1.0)
            
            obs, rewards, dones, extras = vec_env.step(actions)
            total_reward += rewards.sum().item()
            episode_length += 1
            
            if dones.any():
                print(f"Episode finished at step {step + 1}")
                break
        
        print(f"\nResults:")
        print(f"  Episode length: {episode_length}")
        print(f"  Total reward: {total_reward:.3f}")
        
    else:
        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Environments: {args.num_envs}")
        print(f"  Device: {args.device}")
        print("-" * 60)
        
        try:
            runner.learn(
                num_learning_iterations=args.max_iterations,
                init_at_random_ep_len=True,
            )
            print("\n" + "=" * 60)
            print("Training complete!")
            print(f"  Logs saved to: {log_dir}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            print(f"Logs saved to: {log_dir}")
        
        finally:
            # Cleanup
            vec_env.close()
            
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.finish()


if __name__ == "__main__":
    main()

