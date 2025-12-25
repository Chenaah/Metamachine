"""
Train RL Policy with Stable Baselines 3

Simple example using the SB3Trainer utility for easy training setup.
Automatically handles logging, checkpointing, and reward component tracking.

Usage:
    # Train a new policy
    python train_rl_policy_sb3.py
    python train_rl_policy_sb3.py --timesteps 500000
    python train_rl_policy_sb3.py --config modular_quadruped

    # Play/visualize a trained policy
    python train_rl_policy_sb3.py --play ./logs/my_experiment
    python train_rl_policy_sb3.py --play ./logs/my_experiment --checkpoint 200000
    python train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot
"""

import argparse
import os

# Global defaults
DEFAULT_CONFIG = "example_three_modules"
DEFAULT_SEED = 42
DEFAULT_TIMESTEPS = 1000000
DEFAULT_EXP_NAME = "Train 3 Modules"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL Policy with Stable Baselines 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python train_rl_policy_sb3.py
    
    # Train with custom settings
    python train_rl_policy_sb3.py --config modular_quadruped --timesteps 500000
    
    # Play/visualize a trained policy
    python train_rl_policy_sb3.py --play ./logs/my_experiment
    
    # Play specific checkpoint
    python train_rl_policy_sb3.py --play ./logs/my_experiment --checkpoint 200000
    
    # Deploy to real robot
    python train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot
        """
    )
    
    # Training arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Config name or path (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS})"
    )
    parser.add_argument(
        "--exp-name", "-n",
        type=str,
        default=DEFAULT_EXP_NAME,
        help=f"Experiment name (default: {DEFAULT_EXP_NAME})"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="CrossQ",
        help="RL algorithm: CrossQ, SAC, PPO, TD3, TQC (default: CrossQ)"
    )
    
    # Play mode arguments
    parser.add_argument(
        "--play", "-p",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Play/visualize a trained policy from log directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load: 'latest', 'final', 'best', or step number"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to play (0 for infinite)"
    )
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Deploy to real robot instead of simulation"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set environment variables
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # =========================================================================
    # Play Mode
    # =========================================================================
    if args.play:
        print("=" * 60)
        print("Play Mode - Visualize Trained Policy")
        print("=" * 60)
        
        from metamachine.utils.sb3_utils import play_checkpoint
        
        play_checkpoint(
            log_dir=args.play,
            checkpoint=args.checkpoint,
            num_episodes=args.num_episodes,
            render_mode="viewer" if not args.real_robot else "none",
            real_robot=args.real_robot,
            deterministic=True,
            verbose=True,
        )
        return
    
    # =========================================================================
    # Training Mode
    # =========================================================================
    print("=" * 60)
    print("Training Mode - SB3Trainer")
    print("=" * 60)
    
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    from metamachine.utils.sb3_utils import SB3Trainer
    
    # Load config (by name or path)
    if os.path.exists(args.config):
        cfg = ConfigRegistry.create_from_file(args.config)
    else:
        cfg = ConfigRegistry.create_from_name(args.config)
    
    # Create environment
    env = MetaMachine(cfg)
    
    # Train with SB3Trainer
    trainer = SB3Trainer(
        env,
        algorithm=args.algorithm,
        exp_name=args.exp_name,
        seed=args.seed,
        checkpoint_freq=100000,
    )
    trainer.learn(total_timesteps=args.timesteps)
    trainer.save()
    
    print(f"\nTraining complete! Logs saved to: {trainer.log_dir}")
    print(f"To visualize: python train_rl_policy_sb3.py --play {trainer.log_dir}")


if __name__ == "__main__":
    main()
