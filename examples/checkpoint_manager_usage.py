"""
Examples of using the CheckpointManager

This file demonstrates various ways to use the checkpoint management system
in MetaMachine for downloading, caching, and managing model checkpoints.
"""

from metamachine.utils import CheckpointManager, get_checkpoint, register_model, print_models


# Example 1: Using the singleton instance (simplest)
def example_simple_usage():
    """Simple usage with convenience functions."""
    print("=" * 80)
    print("Example 1: Simple Usage")
    print("=" * 80)
    
    # List all registered models
    print_models()
    
    # Download a model by name (uses default manager)
    try:
        model_path = get_checkpoint("example_policy")
        print(f"\nModel downloaded to: {model_path}")
    except Exception as e:
        print(f"Error: {e}")


# Example 2: Creating a custom manager instance
def example_custom_manager():
    """Using a custom CheckpointManager instance."""
    print("\n" + "=" * 80)
    print("Example 2: Custom Manager")
    print("=" * 80)
    
    # Create a custom manager with custom cache directory
    manager = CheckpointManager(cache_dir="/tmp/my_models")
    
    # Register a new model
    manager.register_model(
        name="my_custom_model",
        url="https://example.com/my_model.pkl",
        config="basic_quadruped",
        md5="abc123...",  # Optional MD5 for verification
        description="My custom trained policy",
    )
    
    # List models
    models = manager.list_models()
    print(f"\nRegistered models: {list(models.keys())}")
    
    # Get checkpoint (will download if not cached)
    try:
        model_path = manager.get_checkpoint("my_custom_model")
        print(f"Model path: {model_path}")
    except Exception as e:
        print(f"Error: {e}")


# Example 3: Downloading from arbitrary URLs
def example_download_from_url():
    """Download models from arbitrary URLs."""
    print("\n" + "=" * 80)
    print("Example 3: Download from URL")
    print("=" * 80)
    
    manager = CheckpointManager()
    
    # Download from a direct URL
    try:
        model_path = manager.download_from_url(
            url="https://example.com/some_model.pkl",
            filename="my_downloaded_model",  # Custom filename
            show_progress=True,
        )
        print(f"Downloaded to: {model_path}")
    except Exception as e:
        print(f"Error: {e}")


# Example 4: Cache management
def example_cache_management():
    """Managing the checkpoint cache."""
    print("\n" + "=" * 80)
    print("Example 4: Cache Management")
    print("=" * 80)
    
    manager = CheckpointManager()
    
    # Show cache information
    manager.print_cache_info()
    
    # Get cache info as dictionary
    cache_info = manager.get_cache_info()
    print(f"\nCache has {cache_info['num_files']} files")
    print(f"Total size: {cache_info['total_size_mb']:.2f} MB")
    
    # Clear cache (requires confirmation)
    # manager.clear_cache(confirm=True)


# Example 5: Model registry management
def example_registry_management():
    """Managing the model registry."""
    print("\n" + "=" * 80)
    print("Example 5: Registry Management")
    print("=" * 80)
    
    manager = CheckpointManager()
    
    # Add multiple models
    models_to_add = [
        {
            "name": "quadruped_walk",
            "url": "https://storage.example.com/quadruped_walk.pkl",
            "config": "basic_quadruped",
            "description": "Walking policy for quadruped",
        },
        {
            "name": "tripod_gait",
            "url": "https://storage.example.com/tripod_gait.pkl",
            "config": "example_three_modules",
            "md5": "def456...",
            "description": "Tripod gait policy",
        },
    ]
    
    for model in models_to_add:
        manager.register_model(**model)
    
    # List all models
    manager.print_models()
    
    # Get info for specific model
    info = manager.get_model_info("quadruped_walk")
    print(f"\nModel info: {info}")
    
    # Unregister a model (optionally delete cached file)
    manager.unregister_model("tripod_gait", delete_file=True)


# Example 6: Verifying checksums
def example_checksum_verification():
    """Using MD5 checksums for file verification."""
    print("\n" + "=" * 80)
    print("Example 6: Checksum Verification")
    print("=" * 80)
    
    manager = CheckpointManager()
    
    # Register model with MD5 checksum
    manager.register_model(
        name="verified_model",
        url="https://example.com/verified_model.pkl",
        md5="1234567890abcdef1234567890abcdef",
        description="Model with MD5 verification",
    )
    
    # When downloading, checksum will be automatically verified
    try:
        model_path = manager.get_checkpoint("verified_model")
        print(f"Model verified and saved to: {model_path}")
    except ValueError as e:
        print(f"Checksum verification failed: {e}")
    except Exception as e:
        print(f"Error: {e}")


# Example 7: Integration with policy loading
def example_policy_loading():
    """Complete example of loading a policy for inference."""
    print("\n" + "=" * 80)
    print("Example 7: Policy Loading for Inference")
    print("=" * 80)
    
    try:
        from capyrl import CrossQ
    except ImportError:
        print("CapyRL not installed. Skipping this example.")
        return
    
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    
    # Get checkpoint
    manager = CheckpointManager()
    
    try:
        model_path = manager.get_checkpoint("example_policy")
        
        # Load the policy
        model = CrossQ.load_pkl(str(model_path), env=None, device="cpu")
        print("Policy loaded successfully!")
        
        # Get recommended config from model info
        model_info = manager.get_model_info("example_policy")
        config_name = model_info.get("config", "basic_quadruped")
        
        # Create environment
        cfg = ConfigRegistry.create_from_name(config_name)
        env = MetaMachine(cfg)
        
        print(f"Environment created with config: {config_name}")
        print("Ready for inference!")
        
    except Exception as e:
        print(f"Error: {e}")


# Example 8: Custom registry file
def example_custom_registry():
    """Using a custom registry configuration."""
    print("\n" + "=" * 80)
    print("Example 8: Custom Registry")
    print("=" * 80)
    
    # Define custom registry
    custom_registry = {
        "model_v1": {
            "url": "https://myserver.com/v1.pkl",
            "config": "custom_config",
            "description": "Version 1 of my model",
        },
        "model_v2": {
            "url": "https://myserver.com/v2.pkl",
            "config": "custom_config",
            "md5": "abcdef...",
            "description": "Version 2 of my model",
        },
    }
    
    # Create manager with custom registry
    manager = CheckpointManager(registry=custom_registry)
    
    # Now only custom models are available
    manager.print_models()


if __name__ == "__main__":
    # Run all examples
    print("CheckpointManager Examples")
    print("=" * 80)
    
    # Note: Most examples will fail because the URLs are not real
    # This is just for demonstration purposes
    
    example_simple_usage()
    # example_custom_manager()
    # example_download_from_url()
    # example_cache_management()
    # example_registry_management()
    # example_checksum_verification()
    # example_policy_loading()
    # example_custom_registry()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)
