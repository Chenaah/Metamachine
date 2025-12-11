
"""
Train a transformer policy using CapyFormer and run inference.

Requires: pip install git+https://github.com/Chenaah/CapyFormer.git
"""

import pickle
import time

try:
    from capyformer import Trainer, TrajectoryDataset
except ImportError:
    raise ImportError("Please install capyformer: pip install git+https://github.com/Chenaah/CapyFormer.git")

from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine

# ============ Configuration ============
ROLLOUT_PATH = "rollouts.pkl"
MODEL_SAVE_PATH = "./models/my_model"
LOG_DIR = "./debug"
CONTEXT_LEN = 20
N_EPOCHS = 1000
BATCH_SIZE = 128
# =======================================


class ThreeModulesController(TrajectoryDataset):
    def _setup_dataset(self, dataset_config):
        rollout_data = pickle.load(open(ROLLOUT_PATH, "rb"))
        self.trajectories = rollout_data["trajectories"]
        self.input_keys = ["env_observations"]
        self.target_key = "actions"


def train():
    """Train the transformer policy."""
    traj_dataset = ThreeModulesController({"val_split": 0.1}, CONTEXT_LEN)
    
    trainer = Trainer(
        traj_dataset,
        log_dir=LOG_DIR,
        use_action_tanh=False,
        shared_state_embedding=False,
        n_blocks=3,
        h_dim=256,
        n_heads=1,
        batch_size=BATCH_SIZE,
        validation_freq=1000,
        action_is_velocity=True
    )
    trainer.learn(n_epochs=N_EPOCHS)
    trainer.save(MODEL_SAVE_PATH)
    return trainer


def run_inference(trainer):
    """Run inference with the trained policy."""
    policy = trainer.get_inference()

    cfg = ConfigRegistry.create_from_name("example_three_modules")
    cfg.control.default_dof_pos = [0, 0, 0]
    env = MetaMachine(cfg)
    env.render_mode = "viewer"
    obs, _ = env.reset(seed=123)
    policy.reset()

    for step in range(1000):
        t0 = time.time()

        action = policy.step({'env_observations': obs})
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        # Maintain real-time simulation speed
        elapsed = time.time() - t0
        sleep_time = max(0, env.cfg.control.dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        if done or truncated:
            print(f"Episode ended at step {step}, reward: {reward:.3f}")
            break

    print("Inference completed!")


if __name__ == "__main__":
    trainer = train()
    run_inference(trainer)
