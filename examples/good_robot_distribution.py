# custom obs?

import random

import numpy as np
from capyrl import CrossQ

from metamachine.environments.components.reward import (
    RewardComponent,
    register_component,
)
from metamachine.environments.components.state import register_observation_component
from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine
from metamachine.robot_factory.modular_legs.meta_designer import ModularLegs
from metamachine.robot_factory.modular_legs.morphology import (
    DockPosition,
    ModuleConnection,
)

np.random.seed(233)


def sample_morphology():
    robot_designer = ModularLegs()
    robot_designer.reset()
    pipe = []
    
    for _ in range(4):
        module_id = random.choice(robot_designer.get_available_module_ids())
        print(f"Available module IDs: {robot_designer.get_available_module_ids()}")
        print(f"Selected module ID: {module_id}")
        available_docks = robot_designer.get_available_docks(module_id)
        if not available_docks:
            print(f"No available docks for module {module_id}, skipping...")
            continue
            
        parent_dock = random.choice(available_docks)
        child_dock = random.choice(list(DockPosition)[0:9])  # Only use first 9 positions for child
        orientation = random.choice(robot_designer.get_available_rotation_ids(parent_dock.value, child_dock.value))
        connection = ModuleConnection(
            parent_module_id=module_id,
            parent_dock=parent_dock,
            child_dock=child_dock,
            orientation=orientation
        )
        robot_designer.add_module(connection)
        pipe.extend([module_id, parent_dock.value, child_dock.value, orientation])

    return pipe







# # Register a custom energy component
def global_state(state):
                            # state.last_action
    # state.mj_data
    obs = np.concatenate((  state.accurate_pos_world,
                            state.accurate_vel_world,
                            state.projected_gravity,
                            state.ang_vel_body,
                            state.dof_pos,
                            state.dof_vel,
                            state.last_action
                            ))
    return obs

register_observation_component('global_state', global_state)


class GlobalWalk(RewardComponent):
    """Custom component that rewards energy-efficient movement."""
    
    def calculate(self, state, calculator) -> float:
        
        vel = state.vel_world[0:2]
        
        return vel[0]  # Forward velocity only
# Register your custom component
register_component('global_walk', GlobalWalk)


EXP_NAME = "globalwalk_test"

cfg = ConfigRegistry.create_from_name("quadruped_pose_opt")
cfg.logging.experiment_name = EXP_NAME

#
cfg.randomization.init_joint_pos.enabled = False

cfg.task.reward_components = [{'name': 'global_walk', 'type': "global_walk"}]

cfg.observation.components = [{'name': 'global_state'}]

cfg.initialization.randomize_orientation = False


for _ in range(100):
    morphology = sample_morphology()
    print("Sampled Morphology:", morphology)

    cfg.morphology.configuration = morphology

    env = MetaMachine(cfg)
    o, _, = env.reset()
    print("obs :", o)
    model = CrossQ(env, log_dir=env._log_dir, exp_name=EXP_NAME,
                    device="cuda:0"
                    )
    model.learn(total_timesteps=1000000)