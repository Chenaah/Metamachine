"""
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

import copy
import os
from collections import defaultdict

# Set MuJoCo rendering backend with fallback for headless environments
try:
    # Try EGL first (best for headless with GPU)
    os.environ["MUJOCO_GL"] = "egl"
    # Also set PyOpenGL platform for consistency
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
except Exception:
    # Fallback to osmesa for CPU-only rendering
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Note: optimize_pose is imported conditionally when needed to avoid JAX dependency issues
import pdb
from typing import Any, Optional, Union

# from gymnasium.envs.mujoco import MujocoEnv
import cv2
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf

from metamachine.robot_factory.factory_registry import (
    get_default_draft_model_cfg,
    get_default_fine_model_cfg,
)

from .. import robot_factory
from ..robot_factory.core.xml_compiler import XMLCompiler
from ..utils.math_utils import (
    AverageFilter,
    construct_quaternion,
    quat_rotate,
    quat_rotate_inverse,
    quaternion_from_vectors,
    quaternion_multiply_alt,
    quaternion_to_euler,
    rotate_vector2D,
    wxyz_to_xyzw,
)
from ..utils.validation import is_list_like, is_number
from .base import Base
from .gym.mujoco_env import MujocoEnv

# Try to get CONTROLLER_ROOT_DIR from environment or use default
try:
    from metamachine import METAMACHINE_ROOT_DIR

    ROOT_DIR = METAMACHINE_ROOT_DIR
except ImportError:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class MetaMachine(Base, MujocoEnv):
    """
    MetaMachine Simulation Environment

    A comprehensive robotic simulation environment that provides:
    - Modular robot morphology support with pose optimization
    - Advanced action processing with filtering and multiple control modes
    - Configurable reward systems with multiple components
    - Real-time visualization and video recording capabilities
    - Command-based high-level control interface
    - Comprehensive observation space with temporal stacking

    This environment serves as the main interface for reinforcement learning
    research, evolutionary robotics research, and robotic control experiments
    in the MetaMachine framework.

    Features high-fidelity physics simulation with comprehensive domain
    randomization, sensor simulation, and advanced control features. Currently
    implemented with MuJoCo physics engine.

    Args:
        cfg: OmegaConf configuration object containing all environment parameters

    Example:
        >>> from metamachine.environments.configs.config_registry import ConfigRegistry
        >>> from metamachine.environments.env_sim import MetaMachine
        >>> cfg = ConfigRegistry.create_from_name("basic_quadruped")
        >>> env = MetaMachine(cfg)
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize simulation environment.

        Args:
            cfg: Modern configuration object with simulation parameters
        """
        self.cfg = cfg

        # Validate simulation config
        self._validate_simulation_config(cfg)
        self._setup_logging()
        self._update_pose_cfg(cfg)

        # Initialize base class
        super().__init__(cfg)

        # Initialize simulation components
        self._initialize_environment(cfg)

        # Setup simulation-specific state
        self._setup_simulation_state()

    def _update_pose_cfg(self, cfg: Any) -> None:

        self._pose_setter = {
            "init_pos": lambda v: setattr(cfg.initialization, "init_pos", v),
            "init_quat": lambda v: setattr(cfg.initialization, "init_quat", v),
            "default_dof_pos": lambda v: setattr(cfg.control, "default_dof_pos", v),
            "forward_vec": lambda v: setattr(cfg.observation, "forward_vec", v),
            "projected_forward": lambda v: setattr(
                cfg.observation, "projected_forward_vec", v
            ),
            "projected_upward": lambda v: setattr(
                cfg.observation, "projected_upward_vec", v
            ),
        }

        if hasattr(cfg, "pose_optimization") and cfg.pose_optimization.enabled:
            # Perform pose optimization if enabled
            if cfg.pose_optimization.load_pose is None:
                self.pose_dict = self._optimize_pose()

                # Save optimized pose parameters
                pose_cfg = OmegaConf.create(self.pose_dict)
                if self._log_dir:  # Check that _log_dir is not None
                    pose_cfg_file = os.path.join(self._log_dir, "optimized_pose.yaml")
                    with open(pose_cfg_file, "w") as fp:
                        OmegaConf.save(config=pose_cfg, f=fp.name)
            else:
                pose_cfg_file = cfg.pose_optimization.load_pose
                self.pose_dict = OmegaConf.load(pose_cfg_file)
                print(f"Loaded pose parameters from {pose_cfg_file}")

            # Ensure pose_dict is a dict-like object
            if hasattr(self.pose_dict, "items"):
                pose_items = self.pose_dict.items()
            else:
                # If pose_dict is a tuple/other type, handle appropriately
                pose_items = []

            for key, value in pose_items:
                if key in self._pose_setter:
                    self._pose_setter[key](value)
                else:
                    print(f"Warning: Unrecognized pose parameter '{key}'")

    def _validate_simulation_config(self, cfg: Any) -> None:
        """Validate simulation configuration."""
        if not hasattr(cfg, "simulation"):
            raise ValueError("Missing 'simulation' section in config")

        required_sim_fields = ["mj_dt"]
        missing = [f for f in required_sim_fields if not hasattr(cfg.simulation, f)]
        if missing:
            raise ValueError(f"Missing simulation config fields: {missing}")

    def _initialize_environment(self, cfg: Any) -> None:
        """Initialize MuJoCo simulation components."""
        self.sim_cfg = cfg.simulation

        if hasattr(cfg, "morphology") and cfg.morphology.asset_file is not None:
            # Load and compile robot asset
            self._load_robot_asset()
        elif hasattr(cfg, "morphology") and cfg.morphology.configuration is not None:
            self._load_robot_asset_from_morphology(
                cfg.morphology.robot_type, cfg.morphology.configuration
            )
        else:
            raise ValueError("No robot asset file or morphology provided")

        # Setup MuJoCo environment
        self._setup_mujoco()

        # Initialize terrain system
        self._setup_terrain()

    def _optimize_pose(self) -> Any:
        """Optimize robot pose using the specified optimization method."""
        if (
            not hasattr(self.cfg, "pose_optimization")
            or not self.cfg.pose_optimization.enabled
        ):
            return {}
        else:
            # Import optimize_pose only when needed to avoid JAX dependency issues
            try:
                from metamachine.robot_factory.pose_optimizer import optimize_pose
            except ImportError as e:
                raise ImportError(
                    f"JAX and MuJoCo-MJX are required for pose optimization functionality. "
                    f"Install with: pip install 'metamachine[jax]' or pip install jax jaxlib mujoco-mjx\n"
                    f"Original error: {e}"
                ) from e

            print("Optimizing robot pose...")
            # Implement optimization logic here
            # This could involve running a physics simulation, adjusting joint angles, etc.
            self._load_draft_robot_asset_from_morphology(
                self.cfg.morphology.robot_type,  # type: ignore
                self.cfg.morphology.configuration,  # type: ignore
            )

            pose_dict = optimize_pose(
                self._draft_robot_instance,
                drop_steps=self.cfg.pose_optimization.drop_steps,
                move_steps=self.cfg.pose_optimization.move_steps,
                optimization_type=self.cfg.pose_optimization.optimization_type,
                enable_progress_bar=True,
                spine_assumption=False,
                seed=0,
                log_dir=self._log_dir,
            )
            return pose_dict

    def _load_robot_asset_from_morphology(
        self, robot_type: Any, morphology: Any
    ) -> None:
        """Load robot asset from morphology using the new factory system."""
        import tempfile
        import os
        
        # Get the factory using the new registry system
        factory = robot_factory.get_robot_factory(
            robot_type,
            sim_cfg=self.cfg.simulation,  # type: ignore
            log_dir=self._log_dir,  # Pass the environment's log directory
            **get_default_fine_model_cfg(robot_type),
        )
        if factory is None:
            raise ValueError(f"Unknown robot factory type: {robot_type}")

        # Create robot using the new factory interface
        robot = factory.create_robot(morphology=morphology, log_dir=self._log_dir)

        # Validate the robot
        is_valid, errors = robot.validate()
        if not is_valid:
            print(f"Warning: Robot validation failed: {errors}")

        # Store robot instance for potential future use
        self._robot_instance = robot
        
        # Check if restructure is enabled (for lego_legs or other graph-based robots)
        restructure = getattr(self.cfg.morphology, "restructure", False)
        restructure_qpos = getattr(self.cfg.morphology, "restructure_qpos", None)
        
        if restructure and hasattr(robot, "save"):
            # Save to temp file with restructuring
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                temp_xml_path = f.name
            
            # Convert restructure_qpos from OmegaConf if needed
            if restructure_qpos is not None:
                from omegaconf import DictConfig, ListConfig, OmegaConf
                if isinstance(restructure_qpos, (DictConfig, ListConfig)):
                    restructure_qpos = OmegaConf.to_container(restructure_qpos, resolve=True)
            
            # Save with restructure
            saved_path = robot.save(
                temp_xml_path,
                restructure=True,
                restructure_qpos=restructure_qpos,
            )
            
            # Read the restructured XML
            with open(saved_path, 'r') as f:
                self.xml_string = f.read()
            
            # Create XMLCompiler from the restructured file
            self.xml_compiler = XMLCompiler(saved_path)
        else:
            # Get the XML string directly (no restructure)
            self.xml_string = robot.get_xml_string()
            
            # Create XMLCompiler from the generated XML for randomization support
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(self.xml_string)
                temp_xml_path = f.name
            
            try:
                self.xml_compiler = XMLCompiler(temp_xml_path)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_xml_path):
                    os.unlink(temp_xml_path)

        self.xml_compiler.save(os.path.join(self._log_dir, "robot_debug.xml"))
        
        # Setup mass range if mass randomization is enabled
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled:
            # Get percentage from new or old config
            mass_percentage = mass_cfg.get("percentage", None)
            if mass_percentage is None:
                mass_percentage = self.sim_cfg.get("random_mass_percentage", 0.1)
            
            self.mass_range = self.xml_compiler.get_mass_range(mass_percentage)

    def _load_draft_robot_asset_from_morphology(
        self, robot_type: Any, morphology: Any
    ) -> None:
        """Load robot asset from morphology using the new factory system."""
        # Get the factory using the new registry system
        draft_factory = robot_factory.get_robot_factory(
            robot_type,
            sim_cfg=self.cfg.simulation,  # type: ignore
            log_dir=self._log_dir,  # Pass the environment's log directory
            **get_default_draft_model_cfg(robot_type),
        )

        if draft_factory is None:
            raise ValueError(f"Unknown robot factory type: {robot_type}")

        # # Prepare configuration for robot creation
        # robot_config = {
        #     'sim_cfg': OmegaConf.to_container(self.sim_cfg) if self.sim_cfg else None
        # }

        # Create robot using the new factory interface
        draft_robot = draft_factory.create_robot(morphology=morphology, log_dir=self._log_dir)

        # Validate the robot
        is_valid, errors = draft_robot.validate()
        if not is_valid:
            print(f"Warning: Robot validation failed: {errors}")

        # Get the XML string
        self.draft_xml_string = draft_robot.get_xml_string()

        # Store robot instance for potential future use
        self._draft_robot_instance = draft_robot

    def get_robot_instance(self) -> Any:
        """Get the robot instance if available (new factory system only)."""
        return getattr(self, "_robot_instance", None)

    def _load_robot_asset(self) -> None:
        """Load and compile robot asset."""
        asset_files = self.cfg.morphology.asset_file  # type: ignore
        self.randomize_asset = is_list_like(asset_files)

        asset_file = (
            np.random.choice(asset_files) if self.randomize_asset else asset_files
        )

        xml_path = os.path.join(ROOT_DIR, "assets", "robots", asset_file)

        # Initialize XML compiler with modifications
        self.xml_compiler = XMLCompiler(xml_path)
        self.xml_compiler.torque_control()
        self.xml_compiler.update_timestep(self.sim_cfg.mj_dt)

        # Apply asset modifications
        if self.sim_cfg.get("pyramidal_cone", False):
            self.xml_compiler.pyramidal_cone()

        # Check randomization config (new style: cfg.randomization.mass)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled:
            # Get percentage from new or old config
            mass_percentage = mass_cfg.get("percentage", None)
            if mass_percentage is None:
                mass_percentage = self.sim_cfg.get("random_mass_percentage", 0.1)
            
            self.mass_range = self.xml_compiler.get_mass_range(mass_percentage)

        self.xml_string = self.xml_compiler.get_string()

    def _setup_mujoco(self) -> None:
        """Setup MuJoCo environment."""
        # Calculate simulation parameters
        self.mj_dt = self.sim_cfg.mj_dt
        assert (
            self.cfg.control.dt % self.mj_dt < 1e-9  # type: ignore
        ), f"dt ({self.cfg.control.dt}) must be multiple of mj_dt ({self.mj_dt})"  # type: ignore
        self.frame_skip = int(self.cfg.control.dt / self.mj_dt)  # type: ignore

        # Rendering configuration
        self.render_size = self.sim_cfg.get("render_size", [426, 240])
        # self.render_on = self.sim_cfg.get('render', False)

        # New rendering mode configuration
        self.render_mode = self.sim_cfg.get(
            "render_mode", "none"
        )  # 'none', 'viewer', 'mp4'
        self.video_path = self.sim_cfg.get("video_path", self._log_dir)
        self.video_path = self._log_dir if self.video_path is None else self.video_path

        # Video recording configuration
        self.video_record_interval = self.sim_cfg.get(
            "video_record_interval", 1
        )  # Record every N episodes
        self.video_name_pattern = self.sim_cfg.get(
            "video_name_pattern", "episode_{episode}"
        )  # Naming pattern
        self.video_base_name = self.sim_cfg.get(
            "video_base_name", "robot_video"
        )  # Base name without extension

        # Setup EGL environment if needed
        if self.render_mode == "mp4":
            self._setup_egl_environment()
            self._initialize_video_recording()

        # Initialize MuJoCo
        MujocoEnv.__init__(
            self,
            self.xml_string,
            self.frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=self.render_size[0],
            height=self.render_size[1],
        )

        self._setup_spaces()

        # Restore our render settings after MuJoCo init (it might overwrite them)
        self.render_mode = self.sim_cfg.get(
            "render_mode", "none"
        )  # 'none', 'viewer', 'mp4'
        self.video_fps = self.sim_cfg.get("video_fps", None)
        if self.video_fps is None:
            self.video_fps = 1 / self.cfg.control.dt  # type: ignore

    def _setup_terrain(self) -> None:
        """Setup terrain generation system."""
        # TODO: Implement terrain system when available
        # Currently no 'Terrain' class exists, only TerrainBuilder which requires arguments
        self.terrain_resetter = None
        if hasattr(self.sim_cfg, "get") and self.sim_cfg.get("terrain"):
            print("Warning: Terrain requested but module unavailable")

    def _setup_egl_environment(self) -> None:
        """Setup EGL environment for headless rendering."""
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        os.environ["PYOPENGL_PLATFORM"] = "egl"

        print("EGL environment configured for headless rendering")

    def _initialize_video_recording(self) -> None:
        """Initialize video recording system."""
        self.video_frames: list[np.ndarray] = []
        self.egl_renderer: Optional[Union[str, Any]] = None
        self.recording_active = False
        print(
            f"Video recording initialized. Recording every {self.video_record_interval} episodes with pattern: {self.video_name_pattern}"
        )

    def _create_egl_renderer(self) -> Any:
        """Create EGL renderer for direct rendering."""
        if self.egl_renderer is None:
            try:
                # Ensure EGL is properly set up
                import os

                old_gl = os.environ.get("MUJOCO_GL", "")
                os.environ["MUJOCO_GL"] = "egl"

                width, height = self.render_size
                self.egl_renderer = mujoco.Renderer(
                    self.model, height=height, width=width
                )
                print(f"EGL renderer created ({width}x{height})")

                # Get preferred camera for rendering
                self.preferred_camera_id = self._get_preferred_camera()

                # Restore previous GL setting if it existed
                if old_gl:
                    os.environ["MUJOCO_GL"] = old_gl

            except Exception as e:
                print(f"Failed to create EGL renderer: {e}")
                # Try alternative: use synthetic frames for headless environments
                print("EGL rendering not available, using synthetic mode")
                self.egl_renderer = (
                    "synthetic"  # Special marker for synthetic rendering
                )
        return self.egl_renderer

    def _cleanup_egl_renderer(self) -> None:
        """Clean up EGL renderer."""
        if hasattr(self, "egl_renderer") and self.egl_renderer is not None:
            if self.egl_renderer != "synthetic":
                try:
                    if hasattr(self.egl_renderer, "close"):
                        self.egl_renderer.close()
                except Exception as e:
                    print(f"Warning: Error closing EGL renderer: {e}")
            self.egl_renderer = None

    def _create_synthetic_frame(self, width: int, height: int) -> np.ndarray:
        """Create a synthetic frame with robot state visualization."""
        from io import BytesIO

        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
        ax.set_aspect("equal")

        # Draw ground
        ground = patches.Rectangle((-2, -1), 4, 0.1, color="brown")
        ax.add_patch(ground)

        # Get robot position
        if hasattr(self, "data") and self.data is not None:
            pos = self.data.qpos[:3]
            self.data.qpos[3:7]

            # Draw robot as a simple circle
            robot = patches.Circle((pos[0], pos[2]), 0.2, color="blue", alpha=0.7)
            ax.add_patch(robot)

            # Draw velocity vector
            if hasattr(self, "data"):
                vel = self.data.qvel[:3] * 0.5  # Scale for visibility
                ax.arrow(
                    pos[0],
                    pos[2],
                    vel[0],
                    vel[2],
                    head_width=0.05,
                    head_length=0.05,
                    fc="red",
                    ec="red",
                )

            # Add text info
            ax.text(-1.8, 2.7, f"Step: {getattr(self, 'step_count', 0)}", fontsize=8)
            ax.text(-1.8, 2.5, f"Pos: ({pos[0]:.2f}, {pos[2]:.2f})", fontsize=8)

        ax.set_title("Robot Simulation (Synthetic View)", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Convert to numpy array
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)

        from PIL import Image

        img = Image.open(buf)
        frame = np.array(img)[:, :, :3]  # Remove alpha channel if present

        plt.close(fig)
        buf.close()

        return frame

    def _capture_frame_egl(self) -> Optional[np.ndarray]:
        """Capture frame using EGL renderer."""
        if self.render_mode != "mp4" or not self.recording_active:
            return None

        renderer = self._create_egl_renderer()

        if renderer == "synthetic":
            # Create synthetic frame
            width, height = self.render_size
            frame = self._create_synthetic_frame(width, height)
        else:
            # Use real EGL renderer with preferred camera
            camera_id = getattr(self, "preferred_camera_id", -1)
            renderer.update_scene(self.data, camera=camera_id)
            pixels = renderer.render()
            # frame = (pixels * 255).astype(np.uint8)
            frame = 1 - pixels
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = self._add_metrics_overlay(frame)

        self.video_frames.append(frame)
        return frame

    def _save_video(self) -> bool:
        """Save collected frames to MP4 video using moviepy."""
        if not self.video_frames:
            print("No frames to save")
            return False

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Generate filename based on pattern
        filename = self._generate_video_filename()

        # Create clip from frames with fps specified
        clip = ImageSequenceClip(list(self.video_frames), fps=self.video_fps)

        # Write video file
        clip.write_videofile(
            filename,
            codec="libx264",
            fps=self.video_fps,
            audio=False,
            logger=None,  # Suppress moviepy logging
        )

        # Clean up
        clip.close()

        file_size = os.path.getsize(filename)
        print(
            f"✓ Video saved: {filename} ({file_size:,} bytes, {len(self.video_frames)} frames)"
        )
        return True

    def _generate_video_filename(self) -> str:
        """Generate video filename based on pattern and episode counter."""
        # Extract base name and extension
        video_path = self.video_path or "output"  # Default if None
        base_name, ext = os.path.splitext(video_path)
        if not ext:
            ext = ".mp4"
            video_dir = video_path
        else:
            video_dir = os.path.dirname(video_path)

        # Generate filename using pattern
        if "{episode}" in self.video_name_pattern:
            filename = str(
                self.video_name_pattern.format(episode=self.episode_counter + 1)
            )
        else:
            # Fallback to pattern as is if no {episode} placeholder
            filename = str(self.video_name_pattern)

        # Ensure it has .mp4 extension
        if not filename.endswith(".mp4"):
            filename += ext

        # Use directory from original video_path if provided
        if video_dir:
            filename = os.path.join(video_dir, filename)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        return filename

    def _should_record_episode(self) -> bool:
        """Check if current episode should be recorded based on interval."""
        if self.video_record_interval is None or self.video_record_interval <= 0:
            return False
        return (
            self.episode_counter + 1
        ) % self.video_record_interval == 0 or self.episode_counter == 0

    def start_video_recording(self) -> None:
        """Start video recording."""
        if self.render_mode == "mp4":
            self.video_frames = []
            self.recording_active = True
            print(f"Video recording started for episode {self.episode_counter}")
        else:
            print("Video recording not available (render_mode must be 'mp4')")

    def stop_video_recording(self) -> bool:
        """Stop video recording and save video."""
        if self.render_mode == "mp4" and self.recording_active:
            self.recording_active = False
            success = self._save_video()
            self._cleanup_egl_renderer()
            return success
        return False

    def _setup_simulation_state(self) -> None:
        """Setup simulation-specific state and tracking."""
        # Robot configuration
        self._parse_robot_parameters()

        # Joint and body setup
        self._setup_robot_model()

        # Control tracking
        self._initialize_control_state()

        # Sensor and data tracking
        self._initialize_sensors()

        # External forces
        self._setup_external_forces()

        # Initialization parameters
        self._setup_initialization_parameters()

        # Initialize viewer for viewer mode
        self._passive_viewer = None
        self._viewer_context_manager = None

    def _parse_robot_parameters(self) -> None:
        """Parse robot-specific parameters."""
        # Get parameters from their correct config locations
        self.theta = getattr(self.cfg.environment, "theta", 0.610865)  # type: ignore
        self.kp = getattr(self.cfg.control, "kp", 8.0)  # type: ignore
        self.kd = getattr(self.cfg.control, "kd", 0.2)  # type: ignore

        control_cfg = self.cfg.control  # type: ignore

        if control_cfg.num_actions is None:
            self.num_act = self.model.nu
        else:
            self.num_act = control_cfg.num_actions
        self.num_envs = self.cfg.environment.num_envs  # type: ignore

    def _setup_robot_model(self) -> None:
        """Setup robot model parameters after MuJoCo initialization."""
        self.num_joint = self.model.nu

        # Extract joint module IDs
        self.jointed_module_ids = sorted(
            [
                int(
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j).replace(
                        "joint", ""
                    )
                )
                for j in self.model.actuator_trnid[:, 0]
            ]
        )

        # Validate action space
        expected_joints = self.num_act * self.num_envs
        if expected_joints != self.num_joint:
            raise ValueError(
                f"Action space mismatch: expected {expected_joints}, got {self.num_joint}"
            )

        # Setup joint and body indices
        self.joint_idx = [
            self.model.joint(f"joint{i}").id for i in self.jointed_module_ids
        ]
        self.joint_geom_idx = [
            self.model.geom(f"left{i}").id for i in self.jointed_module_ids
        ] + [self.model.geom(f"right{i}").id for i in self.jointed_module_ids]
        self.joint_body_idx = [
            self.model.geom(f"left{i}").bodyid.item() for i in self.jointed_module_ids
        ]

        # Setup PD gains with per-joint control support
        self._setup_pd_gains()

    def _setup_pd_gains(self) -> None:
        """Setup PD gains with optional per-joint configuration.
        
        This method supports:
        1. Uniform gains (default): All joints use the same kp/kd
        2. Wheel joints: Specific joints configured as velocity-controlled (kp=0, higher kd)
        3. Fine-grained: Explicit per-joint kp/kd arrays
        
        For velocity-controlled joints (wheels), the action is interpreted as velocity target
        rather than position offset.
        """
        per_joint_cfg = getattr(self.cfg.control, 'per_joint_control', None)
        
        if per_joint_cfg and per_joint_cfg.get('enabled', False):
            # Initialize with global values
            self.kps = np.full(self.num_act, self.kp, dtype=np.float32)
            self.kds = np.full(self.num_act, self.kd, dtype=np.float32)
            self.joint_control_modes = ['position'] * self.num_act
            self.default_dof_vel = np.zeros(self.num_act, dtype=np.float32)
            self.wheel_action_scale = 1.0
            
            # Apply wheel joints configuration
            wheel_cfg = per_joint_cfg.get('wheel_joints', {})
            wheel_indices = wheel_cfg.get('indices', [])
            if wheel_indices:
                wheel_kd = wheel_cfg.get('kd', 2.0)
                default_vel = wheel_cfg.get('default_velocity', 0.0)
                self.wheel_action_scale = wheel_cfg.get('action_scale', 1.0)
                
                for idx in wheel_indices:
                    if idx >= self.num_act:
                        raise ValueError(
                            f"Wheel joint index {idx} >= num_actions {self.num_act}"
                        )
                    self.kps[idx] = 0.0  # No position control for wheels
                    self.kds[idx] = wheel_kd
                    self.joint_control_modes[idx] = 'velocity'
                    self.default_dof_vel[idx] = default_vel
            
            # Override with explicit per-joint config if provided
            if per_joint_cfg.get('per_joint_kp') is not None:
                self.kps = np.array(per_joint_cfg.per_joint_kp, dtype=np.float32)
            if per_joint_cfg.get('per_joint_kd') is not None:
                self.kds = np.array(per_joint_cfg.per_joint_kd, dtype=np.float32)
            if per_joint_cfg.get('per_joint_mode') is not None:
                self.joint_control_modes = list(per_joint_cfg.per_joint_mode)
                # Update default_dof_vel for velocity-controlled joints
                for i, mode in enumerate(self.joint_control_modes):
                    if mode == 'velocity' and i not in wheel_indices:
                        self.default_dof_vel[i] = 0.0
            
            # Store wheel indices for later use
            self.wheel_joint_indices = [
                i for i, mode in enumerate(self.joint_control_modes) 
                if mode == 'velocity'
            ]
        else:
            # Original behavior: uniform gains for all joints
            self.kps = np.full(self.num_act * self.num_envs, self.kp, dtype=np.float32)
            self.kds = np.full(self.num_act * self.num_envs, self.kd, dtype=np.float32)
            self.joint_control_modes = None
            self.default_dof_vel = None
            self.wheel_joint_indices = []
            self.wheel_action_scale = 1.0


    def _initialize_control_state(self) -> None:
        """Initialize control state tracking."""
        # Default DOF positions
        default_dof_pos = self.cfg.control.get("default_dof_pos", 0)  # type: ignore
        if isinstance(default_dof_pos, (list, np.ndarray)):
            self.default_dof_pos = np.array(default_dof_pos)
        else:
            self.default_dof_pos = np.full(self.num_joint, default_dof_pos)

        # Latency simulation tracking
        self.last_pos_sim = self.default_dof_pos.copy()
        self.last_last_pos_sim = self.default_dof_pos.copy()
        self.last_vel_sim = np.zeros(self.num_joint)
        self.last_last_vel_sim = np.zeros(self.num_joint)

        # Other tracking
        self.last_com_pos = np.zeros(3)
        self.episode_counter = 0

    def _initialize_sensors(self) -> None:
        """Initialize sensor data structures."""
        self.sensors: defaultdict[str, Any] = defaultdict(list)

    def _setup_external_forces(self) -> None:
        """Setup external force system."""
        self.external_forces_enabled = self.sim_cfg.get("random_external_force", False)
        if self.external_forces_enabled:
            self.external_force_config = {
                "ranges": self.sim_cfg.random_external_force_ranges,
                "bodies": self.sim_cfg.random_external_force_bodies,
                "positions": self.sim_cfg.random_external_force_positions,
                "directions": self.sim_cfg.random_external_force_directions,
                "durations": self.sim_cfg.random_external_force_durations,
                "interval": self.sim_cfg.random_external_force_interval,
            }
            self.external_force_counter = dict.fromkeys(
                range(len(self.external_force_config["bodies"])), 0
            )

    def _setup_initialization_parameters(self) -> None:
        """Setup robot initialization parameters."""
        # Get initialization config
        self.init_cfg = getattr(self.cfg, "initialization", {})

        # Position and orientation
        self.init_pos = self.init_cfg.get("init_pos", [0, 0, 0.1])
        self.init_joint_pos = self.init_cfg.get("init_joint_pos", 0)
        if self.init_joint_pos is None:
            self.init_joint_pos = self.cfg.control.default_dof_pos  # type: ignore

        if isinstance(self.init_joint_pos, (list, np.ndarray)):
            self.init_joint_pos = np.array(self.init_joint_pos)
        else:
            self.init_joint_pos = np.full(self.num_joint, self.init_joint_pos)
        self.given_init_qpos = self.init_cfg.get("init_qpos")

        # Calculate initial quaternion
        self._calculate_initial_quaternion()

        # Forward vector
        forward_vec = self.cfg.observation.get("forward_vec")  # type: ignore
        self.forward_vec = np.array(forward_vec) if forward_vec else None

    def _calculate_initial_quaternion(self) -> None:
        """Calculate initial robot orientation quaternion."""
        init_quat_cfg = self.init_cfg.get("init_quat", "x")
        self.theta = 0.610865  # Default theta value
        lleg_vec = np.array([0, np.cos(self.theta), np.sin(self.theta)])

        if is_list_like(init_quat_cfg):
            if len(init_quat_cfg) == 4:
                self.init_quat = np.array(init_quat_cfg)
            elif len(init_quat_cfg) == 3:
                self.init_quat = quaternion_from_vectors(
                    lleg_vec, np.array(init_quat_cfg)
                )
            else:
                raise ValueError("init_quat list must have 3 or 4 elements")
        elif init_quat_cfg == "x":
            self.init_quat = quaternion_from_vectors(lleg_vec, np.array([1, 0, 0]))
        elif init_quat_cfg == "y":
            self.init_quat = quaternion_from_vectors(lleg_vec, np.array([0, 1, 0]))
        else:
            raise ValueError("init_quat must be list of 3/4 elements or 'x'/'y'")

    def _reset_external_forces(self) -> None:
        """Reset external force applications."""
        self.data.qfrc_applied = np.zeros(self.model.nv)
        if hasattr(self, "external_force_counter"):
            self.external_force_counter = dict.fromkeys(self.external_force_counter, 0)

    def _apply_force(
        self, force: float, body: str, position: list[float], direction: list[float]
    ) -> None:
        """Apply external force to specified body.

        Args:
            force: Force magnitude
            body: Body name
            position: Force application point in local coordinates
            direction: Force direction in global coordinates
        """
        body_id = self.model.body(body).id
        rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
        body_pos = self.data.xpos[body_id]

        # Transform to global coordinates
        force_global = np.array(direction) * force
        point_global = body_pos + rotation_matrix @ np.array(position)

        # Apply force
        torque = np.zeros(3)
        qfrc_result = np.zeros(len(self.data.qvel))
        mujoco.mj_applyFT(
            self.model,
            self.data,
            force_global,
            torque,
            point_global,
            body_id,
            qfrc_result,
        )
        self.data.qfrc_applied = qfrc_result

    def _perform_action(self, action: np.ndarray) -> dict[str, Any]:
        """Execute action with comprehensive control pipeline.

        Args:
            action: Processed action from action processor

        Returns:
            Action execution information
        """

        # Apply external forces
        if self.external_forces_enabled:
            self._handle_external_forces()

        # Validate frame skip for latency
        latency_scheme = self.sim_cfg.get("latency_scheme", -1)
        if latency_scheme >= 0 and self.frame_skip % 2 != 0:
            raise ValueError("frame_skip must be even for latency simulation")

        # Prepare control targets (separate position/velocity, add noise if enabled)
        pos, vel = self._prepare_control_targets(action)

        # Record positions before action
        pos_before = self._record_positions()

        # Execute control with latency simulation
        self._execute_control(pos, vel, latency_scheme)

        # Capture frame for video recording if needed
        self.render()

        # Update latency tracking
        self._update_latency_tracking(pos, vel)

        # Record positions after action
        pos_after = self._record_positions()

        return self._create_action_info(pos_before, pos_after)

    def _handle_external_forces(self) -> None:
        """Handle external force application."""
        config = self.external_force_config

        if self.step_count % config["interval"] == 0:
            for _i, (force_range, body, position, direction) in enumerate(
                zip(
                    config["ranges"],
                    config["bodies"],
                    config["positions"],
                    config["directions"],
                )
            ):
                force = np.random.uniform(*force_range)
                self._apply_force(force, body, position, direction)

        # Update force duration counters
        for i, duration in enumerate(config["durations"]):
            self.external_force_counter[i] += 1
            if self.external_force_counter[i] >= duration:
                self._reset_external_forces()

    def _prepare_control_targets(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prepare position and velocity control targets from actions.
        
        This method:
        1. Separates actions into position/velocity targets based on per-joint control modes
        2. Adds noise if enabled
        
        For position-controlled joints:
            - pos_target = action
            - vel_target = 0
            
        For velocity-controlled joints (wheels):
            - pos_target = 0 (will be multiplied by kp=0)
            - vel_target = action * wheel_action_scale + default_velocity

        Args:
            action: Raw action array from action processor
            
        Returns:
            (pos_target, vel_target): Tuple of position and velocity targets
        """
        pos = action.copy()
        vel = np.zeros_like(pos)

        # Handle per-joint control modes (position vs velocity)
        if self.joint_control_modes is not None:
            for i, mode in enumerate(self.joint_control_modes):
                if mode == 'velocity':
                    # For wheel joints: action is velocity target
                    vel[i] = (action[i] * self.wheel_action_scale 
                              + self.default_dof_vel[i])
                    pos[i] = 0  # Position target is 0 (will be multiplied by kp=0)

        # Add noise if enabled
        if self.sim_cfg.get("noisy_actions", False):
            noise_std = self.sim_cfg.action_noise_std
            pos += self.np_random.normal(0, noise_std, size=pos.shape)
            vel += self.np_random.normal(0, noise_std, size=vel.shape)

        return pos, vel

    def _record_positions(self) -> dict[str, np.ndarray]:
        """Record current robot positions."""
        return {
            "base_pos": self.data.qpos.flat[:2].copy(),
            "general_coords": np.array(
                [self.data.qpos.flat[i : i + 2] for i in self.free_joint_addr]
            ).reshape(-1, 2),
        }

    def _execute_control(
        self, pos: np.ndarray, vel: np.ndarray, latency_scheme: int
    ) -> None:
        """Execute control with latency simulation."""
        if latency_scheme == -1:
            # No latency
            self._pd_control(pos, self.frame_skip, vel)
        elif latency_scheme == 0:
            # One step latency
            half_skip = self.frame_skip // 2
            self._pd_control(self.last_pos_sim, half_skip, self.last_vel_sim)
            self._pd_control(pos, half_skip, vel)
        elif latency_scheme == 1:
            # Two step latency
            half_skip = self.frame_skip // 2
            self._pd_control(self.last_last_pos_sim, half_skip, self.last_last_vel_sim)
            self._pd_control(self.last_pos_sim, half_skip, self.last_vel_sim)
        elif latency_scheme == -2:
            # Fine control with latency
            self._pd_control_fine(
                self.last_pos_sim, self.frame_skip // 2, self.last_vel_sim
            )

    def _update_latency_tracking(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Update latency simulation tracking variables."""
        self.last_last_pos_sim = self.last_pos_sim.copy()
        self.last_pos_sim = pos.copy()
        self.last_last_vel_sim = self.last_vel_sim.copy()
        self.last_vel_sim = vel.copy()

    def _create_action_info(self, pos_before: dict, pos_after: dict) -> dict[str, Any]:
        """Create action execution info dictionary."""
        info = {
            "coordinates": pos_before["base_pos"],
            "next_coordinates": pos_after["base_pos"],
            "coordinates_general": pos_before["general_coords"],
            "next_coordinates_general": pos_after["general_coords"],
        }

        # Add render data if needed
        if getattr(self, "render_on_bg", False):
            rendered_frame = self.render()
            if rendered_frame is not None:
                info["render"] = rendered_frame.transpose(2, 0, 1)

        return info

    def _pd_control(
        self,
        pos_desired: np.ndarray,
        frame_skip: int,
        vel_desired: Optional[np.ndarray] = None,
    ) -> None:
        """Execute PD control for joint positions with mixed control mode support.

        For position-controlled joints:
            torque = kp * (pos_desired - pos_current) + kd * (vel_desired - vel_current)
        
        For velocity-controlled joints (wheels):
            torque = kd * (vel_target - vel_current)  (kp = 0)
            
        Note: The separation of actions into position/velocity targets for wheel
        joints is handled in _prepare_control_targets(), which sets:
        - pos_desired[i] = 0 for wheel joints (multiplied by kp=0)
        - vel_desired[i] = velocity target for wheel joints

        Args:
            pos_desired: Target joint positions (0 for velocity-controlled joints)
            frame_skip: Number of simulation steps
            vel_desired: Target joint velocities (velocity target for wheel joints)
        """
        if vel_desired is None:
            vel_desired = np.zeros_like(pos_desired)

        # Get current joint states
        dof_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_idx]]
        dof_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_idx]]

        # Calculate PD torques
        # For position joints: kp * (pos_error) + kd * (vel_error)
        # For velocity joints: kp=0, so just kd * (vel_target - vel_current)
        torques = self.kps * (pos_desired - dof_pos) + self.kds * (
            vel_desired - dof_vel
        )

        # Apply constraints
        torques = self._apply_torque_constraints(torques, dof_vel)

        # Execute simulation
        self.do_simulation(torques, int(frame_skip))

    def _apply_torque_constraints(
        self, torques: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray:
        """Apply torque and velocity constraints."""
        # Torque-velocity constraints
        if self.sim_cfg.get("tn_constraint", True):
            torque_limits = self._calculate_torque_limits(dof_vel)
            torques = np.clip(torques, -torque_limits, torque_limits)

        # Handle disabled motors
        broken_motors = self.sim_cfg.get("broken_motors")
        if broken_motors is not None:
            torques[broken_motors] = 0

        return torques

    def _calculate_torque_limits(self, dof_vel: np.ndarray) -> np.ndarray:
        """Calculate velocity-dependent torque limits."""
        abs_vel = np.abs(dof_vel)
        torque_limits = np.where(
            abs_vel < 11.5, 12.0, np.clip(-0.656 * abs_vel + 19.541, 0, None)
        )
        return torque_limits

    def _pd_control_fine(
        self,
        pos_desired: np.ndarray,
        frame_skip: int,
        vel_desired: Optional[np.ndarray] = None,
    ) -> None:
        """Execute fine-grained PD control with per-step updates.
        
        Supports mixed control modes (position/velocity) per joint.
        The separation of actions into position/velocity targets is handled
        in _prepare_control_targets().
        """
        if vel_desired is None:
            vel_desired = np.zeros_like(pos_desired)

        for _ in range(int(frame_skip)):
            dof_pos = self.data.qpos[self.model.jnt_qposadr[self.joint_idx]]
            dof_vel = self.data.qvel[self.model.jnt_dofadr[self.joint_idx]]

            torques = self.kps * (pos_desired - dof_pos) + self.kds * (
                vel_desired - dof_vel
            )
            torques = self._apply_torque_constraints(torques, dof_vel)

            self.do_simulation(torques, 1)

    def _get_observable_data(self) -> dict[str, Any]:
        """Get comprehensive observable state data."""
        # Extract basic state
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()

        # Core state components
        state_data = self._extract_core_state(qpos, qvel)

        # Sensor data
        self._update_sensor_readings()
        sensor_data = self._extract_sensor_data()

        # Extended data with noise handling
        extended_data = self._create_extended_state_data({**state_data, **sensor_data})

        # Simulation-specific data
        sim_data = self._extract_simulation_data()

        return {**extended_data, **sim_data}

    def _extract_core_state(self, qpos: np.ndarray, qvel: np.ndarray) -> dict[str, Any]:
        """Extract core robot state information."""
        # Position and orientation
        self.pos_world = qpos[:3]
        quat = wxyz_to_xyzw(qpos[3:7])

        # Joint states
        dof_pos = qpos[self.model.jnt_qposadr[self.joint_idx]]
        dof_vel = qvel[self.model.jnt_dofadr[self.joint_idx]]

        # Velocities
        vel_world = qvel[:3]
        vel_body = quat_rotate_inverse(quat, vel_world)
        ang_vel_body = qvel[3:6]
        ang_vel_world = quat_rotate(quat, ang_vel_body)

        return {
            "pos_world": self.pos_world,
            "quat": quat,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "vel_world": vel_world,
            "vel_body": vel_body,
            "ang_vel_body": ang_vel_body,
            "ang_vel_world": ang_vel_world,
            "qpos": qpos,
            "qvel": qvel,
        }

    def _update_sensor_readings(self) -> None:
        """Update all sensor readings from simulation."""
        sensor_specs = [
            ("quat", 4),
            ("gyro", 3),
            ("vel", 3),
            ("globvel", 3),
            ("back_quat", 4),
            ("back_gyro", 3),
            ("back_vel", 3),
            ("acc", 3),
        ]

        for sensor_type, size in sensor_specs:
            try:
                prefix = "back_imu_" if sensor_type.startswith("back_") else "imu_"
                clean_type = sensor_type.replace("back_", "")

                sensor_data = []
                for module_id in self.jointed_module_ids:
                    sensor_name = f"{prefix}{clean_type}{module_id}"
                    start_addr = self.model.sensor(sensor_name).adr[0]
                    sensor_data.append(
                        self.data.sensordata[start_addr : start_addr + size]
                    )

                self.sensors[sensor_type] = np.array(sensor_data)
            except (KeyError, IndexError):
                continue

    def _extract_sensor_data(self) -> dict[str, Any]:
        """Extract sensor data for observations."""
        sensor_data = {}

        if "quat" in self.sensors:
            sensor_data["quats"] = np.array(
                [wxyz_to_xyzw(q) for q in self.sensors["quat"]]
            )
        if "gyro" in self.sensors:
            sensor_data["gyros"] = self.sensors["gyro"]
        if "acc" in self.sensors:
            sensor_data["accs"] = self.sensors["acc"]

        return sensor_data

    def _create_extended_state_data(self, base_data: dict[str, Any]) -> dict[str, Any]:
        """Create extended data with accurate versions and noise."""
        extended = base_data.copy()

        # Add accurate (noise-free) versions
        for key, value in base_data.items():
            extended[f"accurate_{key}"] = copy.deepcopy(value)

            # Add observation noise if enabled
            if self.sim_cfg.get("noisy_observations", False) and isinstance(
                value, np.ndarray
            ):
                noise_std = self.sim_cfg.obs_noise_std
                extended[key] = value + self.np_random.normal(
                    0, noise_std, size=value.shape
                )

        return extended

    def _extract_simulation_data(self) -> dict[str, Any]:
        """Extract simulation-specific data."""
        sim_data = {
            "mj_data": self.data,
            "mj_model": self.model,
            "adjusted_forward_vec": getattr(
                self, "adjusted_forward_vec", self.forward_vec
            ),
        }

        # Contact information
        sim_data.update(self._extract_contact_data())

        # Additional sensor data
        if "vel" in self.sensors:
            sim_data["vels"] = self.sensors["vel"]
        if "back_vel" in self.sensors:
            sim_data["back_vels"] = self.sensors["back_vel"]
        if "back_quat" in self.sensors:
            sim_data["back_quats"] = np.array(
                [wxyz_to_xyzw(q) for q in self.sensors["back_quat"]]
            )
        if "back_gyro" in self.sensors:
            sim_data["back_gyros"] = self.sensors["back_gyro"]

        # Center of mass data
        sim_data.update(self._calculate_com_data())

        return sim_data

    def _extract_contact_data(self) -> dict[str, Any]:
        """Extract contact information."""
        contacts = [c.geom for c in self.data.contact]
        floor_contacts = [c.geom for c in self.data.contact if 0 in c.geom]

        # Count joint-floor contacts
        joint_floor_count = sum(
            (contact[0] in self.joint_geom_idx) or (contact[1] in self.joint_geom_idx)
            for contact in floor_contacts
        )

        return {
            "contact_geoms": contacts,
            "num_jointfloor_contact": joint_floor_count,
            "contact_floor_geoms": list(
                {geom for pair in floor_contacts for geom in pair if geom != 0}
            ),
            "contact_floor_socks": list(
                {
                    geom
                    for pair in floor_contacts
                    for geom in pair
                    if self.model.geom(geom).name.startswith("sock")
                }
            ),
            "contact_floor_balls": list(
                {
                    geom
                    for pair in floor_contacts
                    for geom in pair
                    if (
                        self.model.geom(geom).name.startswith("left")
                        or self.model.geom(geom).name.startswith("right")
                    )
                }
            ),
        }

    def _calculate_com_data(self) -> dict[str, Any]:
        """Calculate center of mass data."""
        com_pos = np.mean(self.data.xpos[self.joint_body_idx], axis=0)
        com_vel_world = (com_pos - self.last_com_pos) / self.dt
        self.last_com_pos = com_pos.copy()

        return {"com_vel_world": com_vel_world}

    def _reset_robot(self) -> None:
        """Reset robot with domain randomization."""
        # Reset MuJoCo model
        self.reset_model()

        # Reset control tracking
        self.last_pos_sim = self.default_dof_pos.copy()
        self.last_last_pos_sim = self.default_dof_pos.copy()
        self.last_vel_sim = np.zeros(self.num_joint)
        self.last_last_vel_sim = np.zeros(self.num_joint)
        self.last_com_pos = np.zeros(3)

        # Initialize rendering filter
        self.render_lookat_filter = AverageFilter(10)

    def reset_model(self) -> np.ndarray:
        """Reset MuJoCo model with comprehensive domain randomization."""

        # Check if this episode should be recorded
        if self._should_record_episode():
            self.start_video_recording()

        # Pre-reset operations
        self._pre_reset()

        # Handle model reloading if needed
        if self._need_model_reload():
            self._reload_model_with_randomization()

        # Reset external forces
        self._reset_external_forces()

        # Apply domain randomization
        self._apply_domain_randomization()

        # Set initial state
        self._set_initial_state()

        # Post-reset operations
        self._post_reset()

        # Return empty observation as placeholder for parent class compatibility
        return np.array([])

    def _pre_reset(self) -> None:
        """Pre-reset operations."""
        pass

    def _post_reset(self) -> None:
        """Post-reset operations."""
        pass

    def _post_done(self) -> None:
        """Post-done operations after an episode ends."""
        # Stop video recording if active
        if self.render_mode == "mp4" and self.recording_active:
            success = self.stop_video_recording()
            if success:
                print(f"Episode {self.episode_counter} video recording completed")
            else:
                print(f"Episode {self.episode_counter} video recording failed")
        self.episode_counter += 1

    def _need_model_reload(self) -> bool:
        """Check if model needs to be reloaded for randomization."""
        self.randomize_asset = is_list_like(self.cfg.morphology.asset_file)  # type: ignore
        
        # Check randomization config (new style: cfg.randomization.*)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        mass_enabled = randomization_cfg.get("mass", {}).get("enabled", False)
        damping_enabled = randomization_cfg.get("damping", {}).get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        if not damping_enabled:
            damping_enabled = self.sim_cfg.get("randomize_damping", False)
        
        return any(
            [
                mass_enabled,
                damping_enabled,
                self.sim_cfg.get("add_scaffold_walls", False),
                self.randomize_asset,
            ]
        )

    def _reload_model_with_randomization(self) -> None:
        """Reload model with randomization applied."""
        # Asset randomization (for asset-based robots with multiple asset files)
        if self.randomize_asset:
            self._load_robot_asset()
        # For morphology-based robots, regenerate from morphology
        elif not hasattr(self, 'xml_compiler') or self.xml_compiler is None:
            # Morphology-based robot without xml_compiler - skip randomization
            return

        # Get randomization config (new style)
        randomization_cfg = getattr(self.cfg, "randomization", {})
        
        # Mass randomization
        mass_cfg = randomization_cfg.get("mass", {})
        mass_enabled = mass_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not mass_enabled:
            mass_enabled = self.sim_cfg.get("randomize_mass", False)
        
        if mass_enabled and hasattr(self, 'mass_range') and hasattr(self, 'xml_compiler'):
            # Get mass offset from new or old config
            mass_offset = mass_cfg.get("offset", None)
            if mass_offset is None:
                mass_offset = self.sim_cfg.get("mass_offset", 0)
            
            mass_dict = {
                key: np.random.uniform(*value) + mass_offset
                for key, value in self.mass_range.items()
            }
            self.xml_compiler.update_mass(mass_dict)

        # Damping randomization
        damping_cfg = randomization_cfg.get("damping", {})
        damping_enabled = damping_cfg.get("enabled", False)
        
        # Fallback to old style for backward compatibility
        if not damping_enabled:
            damping_enabled = self.sim_cfg.get("randomize_damping", False)
        
        if damping_enabled and hasattr(self, 'xml_compiler'):
            # Get ranges from new or old config
            damping_range = damping_cfg.get("range", None)
            armature_range = damping_cfg.get("armature_range", None)
            
            if damping_range is None:
                damping_range = self.sim_cfg.get("random_damping_range", [0.02, 0.2])
            if armature_range is None:
                armature_range = self.sim_cfg.get("random_armature_range", [0.01, 0.05])
            
            armature = np.random.uniform(*armature_range)
            damping = np.random.uniform(*damping_range)
            self.xml_compiler.update_damping(armature=armature, damping=damping)

        # Scaffold walls
        if self.sim_cfg.get("add_scaffold_walls", False):
            self.xml_compiler.remove_walls()
            angle = quaternion_to_euler(self.init_quat)[0] * 180 / np.pi - 90
            self.xml_compiler.add_walls(transparent=False, angle=angle)

        # Reload MuJoCo environment
        self.reload_model(self.xml_compiler.get_string())

    def _apply_domain_randomization(self) -> None:
        """Apply domain randomization parameters."""
        # PD controller randomization
        randomization_cfg = getattr(self.cfg, "randomization", {})
        pd_cfg = randomization_cfg.get("pd_controller", {})
        if pd_cfg.get("enabled", False):
            self.kp = np.random.uniform(*pd_cfg.get("kp_range", [8.0, 8.0]))
            self.kd = np.random.uniform(*pd_cfg.get("kd_range", [0.2, 0.2]))
            
            # Preserve per-joint control settings when randomizing
            if self.joint_control_modes is not None:
                # Only randomize position-controlled joints
                for i, mode in enumerate(self.joint_control_modes):
                    if mode == 'position':
                        self.kps[i] = self.kp
                        self.kds[i] = self.kd
                    # Velocity-controlled joints keep their original kd
            else:
                # Original behavior: uniform gains
                self.kps = np.full(self.num_act * self.num_envs, self.kp, dtype=np.float32)
                self.kds = np.full(self.num_act * self.num_envs, self.kd, dtype=np.float32)

        # Latency randomization
        if self.sim_cfg.get("random_latency_scheme", False):
            self.sim_cfg.latency_scheme = np.random.randint(0, 2)

        # Friction randomization
        self._randomize_friction()

    def _randomize_friction(self) -> None:
        """Apply friction randomization."""
        randomization_cfg = getattr(self.cfg, "randomization", {})
        friction_cfg = randomization_cfg.get("friction", {})
        if not friction_cfg.get("enabled", False):
            return

        friction_range = friction_cfg.get("range", [0.8, 1.2])

        if is_number(friction_range[0]):
            # Single friction value
            friction = np.random.uniform(*friction_range)
            self.model.geom("floor").friction[0] = friction
            self.model.geom("floor").priority[0] = 10
        else:
            # Separate friction for different components
            stick_friction = np.random.uniform(*friction_range[0])
            ball_friction = np.random.uniform(*friction_range[1])

            self.model.geom("floor").priority[0] = 1
            for module_id in self.jointed_module_ids:
                self.model.geom(f"left{module_id}").friction[0] = ball_friction
                self.model.geom(f"right{module_id}").friction[0] = ball_friction
                self.model.geom(f"stick{module_id}").friction[0] = stick_friction

                for geom_name in [
                    f"left{module_id}",
                    f"right{module_id}",
                    f"stick{module_id}",
                ]:
                    self.model.geom(geom_name).priority[0] = 2

        # Rolling friction
        rolling_cfg = friction_cfg.get("rolling", {})
        if rolling_cfg.get("enabled", False):
            rolling_range = rolling_cfg.get("range", [0.0001, 0.0005])
            roll_friction = np.random.uniform(
                rolling_range[0], rolling_range[1], size=2
            )
            self.model.geom("floor").friction[1:3] = roll_friction

    def _set_initial_state(self) -> None:
        """Set initial robot state with randomization."""
        # Handle orientation randomization
        final_quat = self._get_randomized_orientation()

        # Setup initial position
        self._setup_initial_positions(final_quat)

        # Apply initial state with noise
        qpos = self._apply_initial_noise()
        qvel = self._get_initial_velocities()

        # Set MuJoCo state
        self.set_state(qpos, qvel)

        # Record reset position
        self.reset_pos = self.data.qpos[:2].copy()

    def _get_randomized_orientation(self) -> np.ndarray:
        """Get randomized initial orientation."""
        if self.init_cfg.get("fully_randomize_orientation", False):
            # Fully random orientation
            rand_quat = self.np_random.normal(0, 1, 4)
            return rand_quat / np.linalg.norm(rand_quat)
        elif self.init_cfg.get("randomize_orientation", False):
            # Random rotation around Z-axis
            rotate_angle = self.np_random.uniform(0, 2 * np.pi)
            rand_rotation = construct_quaternion([0, 0, 1], rotate_angle)
            final_quat = quaternion_multiply_alt(self.init_quat, rand_rotation)

            # Update forward vector
            if self.forward_vec is not None:
                self.adjusted_forward_vec = rotate_vector2D(
                    self.forward_vec[:2], -rotate_angle
                )
            return final_quat
        else:
            # No orientation randomization
            if self.forward_vec is not None:
                self.adjusted_forward_vec = self.forward_vec
            return self.init_quat

    def _setup_initial_positions(self, quat: np.ndarray) -> None:
        """Setup initial positions and joint states."""
        # Handle multiple initial positions
        if is_list_like(self.init_pos[0]):
            init_pos_list = copy.deepcopy(self.init_pos)
            quat_list = [quat] + [[1, 0, 0, 0]] * 10  # Default quaternion list

            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == 0:  # mjJNT_FREE
                    qpos_adr = self.model.jnt_qposadr[i]
                    self.init_qpos[qpos_adr : qpos_adr + 3] = init_pos_list.pop(0)
                    self.init_qpos[qpos_adr + 3 : qpos_adr + 7] = quat_list.pop(0)
        else:
            self.init_qpos[:3] = self.init_pos
            self.init_qpos[3:7] = quat

        # Setup free joint addresses
        self.free_joint_addr = [
            self.model.jnt_qposadr[i]
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == 0
        ]

        # Joint position randomization
        randomization_cfg = getattr(self.cfg, "randomization", {})
        dof_cfg = randomization_cfg.get("init_joint_pos", {})
        if dof_cfg.get("enabled", False):
            clip_actions = self.cfg.control.symmetric_limit  # type: ignore
            joint_noise = self.np_random.uniform(
                -clip_actions, clip_actions, self.num_joint
            )
            self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = (
                self.default_dof_pos + joint_noise
            )
            # TODO: consider frozen joints
        else:
            self.init_qpos[self.model.jnt_qposadr[self.joint_idx]] = self.init_joint_pos

    def _apply_initial_noise(self) -> np.ndarray:
        """Apply initial position noise."""
        if self.given_init_qpos is not None:
            return np.array(self.given_init_qpos)

        if self.init_cfg.get("noisy_init", True):
            return np.asarray(
                self.init_qpos + self.np_random.uniform(-0.1, 0.1, size=self.model.nq)
            )
        else:
            return np.asarray(self.init_qpos.copy())

    def _get_initial_velocities(self) -> np.ndarray:
        """Get initial velocities with randomization."""
        init_qvel = getattr(self, "init_qvel", np.zeros(self.model.nv))

        if self.init_cfg.get("randomize_ini_vel", True):
            randomize_vel = self.init_cfg.get("randomize_ini_vel", True)
            if is_list_like(randomize_vel):
                for i, vel_range in enumerate(randomize_vel):
                    init_qvel[i] = self.np_random.uniform(-vel_range, vel_range)
            else:
                init_qvel[:6] = self.np_random.uniform(-1, 1, 6)

        return np.asarray(init_qvel + self.np_random.normal(0, 0.1, self.model.nv))

    def reload_model(self, xml_string: str) -> None:
        """Reload MuJoCo model with new XML."""

        # Preserve render settings
        saved_render_mode = getattr(self, "render_mode", "none")
        saved_video_path = getattr(self, "video_path", "robot_video.mp4")
        saved_video_fps = getattr(self, "video_fps", 20)
        saved_recording_active = getattr(self, "recording_active", False)
        saved_video_frames = getattr(self, "video_frames", [])

        MujocoEnv.__init__(
            self,
            xml_string,
            self.frame_skip,
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode="human" if self.render_mode == "viewer" else "rgb_array",
            width=self.render_size[0],
            height=self.render_size[1],
        )

        # Restore render settings
        self.render_mode = saved_render_mode
        self.video_path = saved_video_path
        self.video_fps = saved_video_fps
        self.recording_active = saved_recording_active
        self.video_frames = saved_video_frames

        # Recreate EGL renderer if needed
        if self.render_mode == "mp4":
            self.egl_renderer = None  # Force recreation
            self.preferred_camera_id = None  # Reset camera preference

    def render(self) -> Optional[Union[np.ndarray, Any]]:  # type: ignore
        """Render environment with different modes: 'none', 'viewer', 'mp4'."""
        if self.render_mode == "none":
            # No rendering
            return None

        elif self.render_mode == "viewer":
            # Use passive viewer with simulation loop
            if self._passive_viewer is None:
                # Initialize passive viewer on first call
                self._viewer_context_manager = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
                assert self._viewer_context_manager is not None
                self._passive_viewer = self._viewer_context_manager.__enter__()  # type: ignore
                print("Passive viewer initialized successfully")

            return None  # type: ignore

        elif self.render_mode == "mp4":
            # EGL rendering for video recording
            return self._capture_frame_egl()

        else:
            print(f"Warning: Unknown render mode '{self.render_mode}'. Using 'none'.")
            return None

    def close(self) -> None:
        """Close environment and cleanup resources."""
        # Stop any active video recording
        if (
            hasattr(self, "render_mode")
            and self.render_mode == "mp4"
            and getattr(self, "recording_active", False)
        ):
            self.stop_video_recording()

        # Cleanup passive viewer
        if (
            hasattr(self, "_viewer_context_manager")
            and self._viewer_context_manager is not None
        ):
            try:  # type: ignore
                self._viewer_context_manager.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self._passive_viewer = None
                self._viewer_context_manager = None

        # Cleanup EGL renderer
        self._cleanup_egl_renderer()

        # Call parent close
        super().close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def _get_available_cameras(self) -> list:
        """Get list of available cameras from the MuJoCo model."""
        if not hasattr(self, "model") or self.model is None:
            return []

        cameras = []
        for i in range(self.model.ncam):
            camera_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if camera_name:
                cameras.append({"id": i, "name": camera_name})
        return cameras

    def _get_preferred_camera(self) -> Optional[Union[str, int]]:
        """Get the preferred camera for rendering, preferring XML-defined cameras."""
        cameras = self._get_available_cameras()

        if not cameras:
            # No cameras defined in XML, use default free camera
            return -1

        # Check if a specific camera is configured
        configured_camera = self.sim_cfg.get("render_camera", None)
        if configured_camera is not None:
            # Try to find camera by name or ID
            if isinstance(configured_camera, str):
                for camera in cameras:
                    if camera["name"] == configured_camera:
                        print(
                            f"Using configured XML camera: {camera['name']} (ID: {camera['id']})"
                        )
                        return int(camera["id"])
                print(f"Warning: Configured camera '{configured_camera}' not found")
            elif isinstance(configured_camera, int):
                if 0 <= configured_camera < len(cameras):
                    camera = cameras[configured_camera]
                    print(
                        f"Using configured XML camera by ID: {camera['name']} (ID: {camera['id']})"
                    )
                    return int(configured_camera)
                print(f"Warning: Configured camera ID {configured_camera} out of range")

        # Prefer cameras with specific names in order of preference
        preferred_names = [
            "follow_camera",
            "main_camera",
            "robot_camera",
            "tracking_camera",
        ]

        for preferred_name in preferred_names:
            for camera in cameras:
                if preferred_name in camera["name"].lower():
                    print(f"Using XML camera: {camera['name']} (ID: {camera['id']})")
                    return int(camera["id"])

        # If no preferred camera found, use the first available camera
        camera = cameras[0]
        print(
            f"Using first available XML camera: {camera['name']} (ID: {camera['id']})"
        )
        return int(camera["id"])

    def _add_metrics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add text overlays with action, reward, and custom metrics."""
        if not getattr(self, "video_show_metrics", True):
            return frame

        # Get current metrics
        # metrics = self._get_current_metrics()
        # np.array2string(self.state.action_history.last_action, precision=2, separator=',', suppress_small=True)
        metrics = {
            "Action": np.array2string(
                self.state.action_history.last_action,
                precision=2,
                separator=",",
                suppress_small=True,
            ),
            "Reward": f"{self.state.reward_history.last_reward:.2f}",
            "Speed": f"{self.state.speed[0]:.2f}",
            "Step": self.step_count,
            "Episode": self.episode_counter + 1,
        }

        cmd_dict = self.state.command_manager.get_commands_dict()
        if cmd_dict:
            for key, value in cmd_dict.items():
                metrics["Cmd:" + key] = (
                    f"{value:.2f}" if isinstance(value, float) else str(value)
                )

        if not metrics:
            return frame

        # Configure text appearance
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)  # White text
        thickness = 1
        line_height = 20
        start_y = 20

        # Draw semi-transparent background for better readability
        overlay = frame.copy()
        bg_height = len(metrics) * line_height + 10
        cv2.rectangle(overlay, (5, 5), (350, bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        for i, (label, value) in enumerate(metrics.items()):
            y_pos = start_y + (i * line_height)
            text = f"{label}: {value}"
            cv2.putText(frame, text, (10, y_pos), font, font_scale, color, thickness)

        return frame

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        Integrates with passive viewer when in 'viewer' mode.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )

        if self.render_mode == "viewer" and self._passive_viewer is not None:
            # Use passive viewer with individual step control
            self.data.ctrl[:] = ctrl  # type: ignore

            for _i in range(n_frames):
                mujoco.mj_step(self.model, self.data)

                # Sync with viewer
                self._passive_viewer.sync()

                # Timing control - sleep for timestep duration
                # time.sleep(self.model.opt.timestep)
        else:
            # Standard simulation without viewer
            self._step_mujoco_simulation(ctrl, n_frames)

    def _step_mujoco_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """
        Step over the MuJoCo simulation (standard implementation).
        """
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        mujoco.mj_rnePostConstraint(self.model, self.data)
