"""
MetaMachine Basic Usage Example
This script demonstrates the basic usage of MetaMachine, a simulation framework
for modular robots designed for reinforcement learning and evolutionary robotics
research. It shows how to:
1. Create and configure an environment
2. Reset the environment
3. Run a simple control loop with actions
4. Monitor environment state and termination conditions
The example uses a basic quadruped configuration and applies a simple
sinusoidal control signal to demonstrate the environment's response.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine

# Create environment configuration
# The "basic_quadruped" config provides a standard 4-legged robot setup
cfg = ConfigRegistry.create_from_name("basic_quadruped")

# Initialize the MetaMachine simulation environment
env = MetaMachine(cfg)
# Set rendering mode to viewer for visual output
env.render_mode = "viewer"

# Reset environment to initial state with fixed seed for reproducibility
env.reset(seed=42)

# Main control loop - run for up to 1000 simulation steps
for step in range(1000):
    # Create action vector (5 DOF for basic quadruped)
    action = np.zeros(5)

    # Apply sinusoidal control to first joint as demonstration
    # This creates a smooth oscillating motion
    action[0] = np.sin(step * 0.1)

    # Step the environment forward one timestep
    # Returns: observation, reward, done, truncated, info
    obs, reward, done, truncated, info = env.step(action)

    # Render the current state of the environment
    env.render()

    # Access current command state (if using command-based control)
    current_commands = env.commands

    # Print progress every 20 steps to monitor training
    if step % 20 == 0:
        print(f"Step {step}: reward={reward:.3f}, done={done}, truncated={truncated}")

    # Check for episode termination
    if done or truncated:
        print(f"Episode ended at step {step}")
        break
