import os
from capyrl import CrossQ
from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine

# Global configuration variables
CONFIG_NAME = "example_three_modules"
SEED = 42
TOTAL_TIMESTEPS = 1000000
EXP_NAME = "Train 3 Modules"

if __name__ == "__main__":
    # Set JAX environment variables
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Create environment configuration
    # The "example_three_modules" config provides a standard 3-legged robot setup
    cfg = ConfigRegistry.create_from_name(CONFIG_NAME)
    
    # Initialize the MetaMachine simulation environment
    env = MetaMachine(cfg)
    
    # Get log directory from environment
    log_dir = env._log_dir
    
    # Train the model
    model = CrossQ(env, log_dir=log_dir, exp_name=EXP_NAME)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)


