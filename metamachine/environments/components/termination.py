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

from enum import Enum

import numpy as np
from omegaconf import OmegaConf

from ...utils.math_utils import quat_rotate_inverse


class TerminationStrategy(Enum):
    BALLANCE = "ballance"
    BALLANCE_UPSIDEDOWN = "ballance_upsidedown"
    BALLANCE_UP = "ballance_up"
    BALLANCE_AUTO = "ballance_auto"
    TORSO_FALL = "torso_fall"
    THREE_FEET = "three_feet"
    BALL_FALL = "ball_fall"
    BODY_CONTACT_FLOOR = "body_contact_floor"


class TerminationChecker:
    """Handles episode termination conditions."""

    def __init__(self, cfg: OmegaConf) -> None:
        """Initialize termination checker.

        Args:
            cfg: Configuration object containing termination settings
        """
        self.cfg = cfg
        self._parse_config(cfg)
        self._setup_termination_handlers()

        # Step tracking for max_episode_steps
        self.current_step = 0

    def _parse_config(self, cfg: OmegaConf) -> None:
        """Parse configuration parameters.

        Args:
            cfg: Configuration object
        """
        # Parse termination conditions from task config
        term_cfg = cfg.task.termination_conditions

        self.termination_strategy = (
            TerminationStrategy(term_cfg.termination_strategy)
            if term_cfg.termination_strategy is not None
            else None
        )

        # Max episode steps termination
        self.max_episode_steps = getattr(term_cfg, "max_episode_steps", 1000)

        # Height threshold termination
        self.height_threshold = getattr(term_cfg, "height_threshold", None)

        # Robot and observation parameters
        self.gravity_vec = np.array(cfg.observation.gravity_vec)
        self.theta = getattr(cfg.environment, "theta", 0.0)  # Updated path
        self.projected_upward_vec = (
            np.array(cfg.observation.projected_upward_vec, dtype=float)
            if cfg.observation.projected_upward_vec is not None
            else None
        )

        # Thresholds (could be moved to config in the future)
        self.balance_threshold = 0.01
        self.orientation_threshold = 0.1

        # Body/geom contact termination - parse names from config
        # Option 1: terminate_on_body_contact - list of body names
        # (terminates if ANY geom belonging to the body contacts floor)
        terminate_bodies = getattr(term_cfg, "terminate_on_body_contact", None)
        if terminate_bodies is not None:
            if isinstance(terminate_bodies, str):
                terminate_bodies = [terminate_bodies]
            self.terminate_on_body_contact = list(terminate_bodies)
        else:
            self.terminate_on_body_contact = None

        # Option 2: terminate_on_geom_contact - list of geom names
        # (terminates if the specific geom contacts floor)
        terminate_geoms = getattr(term_cfg, "terminate_on_geom_contact", None)
        if terminate_geoms is not None:
            if isinstance(terminate_geoms, str):
                terminate_geoms = [terminate_geoms]
            self.terminate_on_geom_contact = list(terminate_geoms)
        else:
            self.terminate_on_geom_contact = None

        # Will be populated by set_model() with geom IDs for the configured bodies/geoms
        self._body_contact_geom_ids = set()

    def _setup_termination_handlers(self) -> None:
        """Set up mapping of termination strategies to their handler functions."""
        self._termination_handlers = {
            TerminationStrategy.BALLANCE: self._check_ballance,
            TerminationStrategy.BALLANCE_UPSIDEDOWN: self._check_ballance_upsidedown,
            TerminationStrategy.BALLANCE_UP: self._check_ballance_up,
            TerminationStrategy.BALLANCE_AUTO: self._check_ballance_auto,
            TerminationStrategy.TORSO_FALL: self._check_torso_fall,
            TerminationStrategy.THREE_FEET: self._check_three_feet,
            TerminationStrategy.BALL_FALL: self._check_ball_fall,
            TerminationStrategy.BODY_CONTACT_FLOOR: self._check_body_contact_floor,
        }

    def reset(self) -> None:
        """Reset the termination checker for a new episode."""
        self.current_step = 0

    def set_model(self, model) -> None:
        """Set up body/geom name to geom ID mapping from MuJoCo model.

        This method should be called after the MuJoCo model is loaded to
        enable body/geom contact termination checking.

        Args:
            model: MuJoCo model object (mujoco.MjModel)
        """
        if self.terminate_on_body_contact is None and self.terminate_on_geom_contact is None:
            return

        self._body_contact_geom_ids = set()

        # Process body names - find all geoms belonging to each body
        if self.terminate_on_body_contact is not None:
            for body_name in self.terminate_on_body_contact:
                try:
                    body_id = model.body(body_name).id
                    # Find all geoms belonging to this body
                    for geom_id in range(model.ngeom):
                        if model.geom_bodyid[geom_id] == body_id:
                            self._body_contact_geom_ids.add(geom_id)
                except KeyError:
                    print(
                        f"Warning: Body '{body_name}' not found in model. "
                        f"Body contact termination will not work for this body."
                    )

        # Process geom names - look up geom IDs directly
        if self.terminate_on_geom_contact is not None:
            for geom_name in self.terminate_on_geom_contact:
                try:
                    geom_id = model.geom(geom_name).id
                    self._body_contact_geom_ids.add(geom_id)
                except KeyError:
                    print(
                        f"Warning: Geom '{geom_name}' not found in model. "
                        f"Geom contact termination will not work for this geom."
                    )

    def step(self) -> None:
        """Increment the step counter."""
        self.current_step += 1

    def check_done(self, state) -> bool:
        """Check if episode should terminate based on current state.

        Args:
            state: Current environment state

        Returns:
            bool: Whether episode should terminate
        """

        # Check height threshold
        if self.height_threshold is not None and hasattr(state, "pos"):
            if state.pos[2] < self.height_threshold:  # z-coordinate below threshold
                return True

        # Check strategy-specific termination
        if self.termination_strategy is None:
            return False

        handler = self._termination_handlers.get(self.termination_strategy)
        if handler is None:
            raise ValueError(
                f"Unknown termination strategy: {self.termination_strategy}"
            )

        return handler(state)

    def _check_ballance(self, state) -> bool:
        return (
            np.dot(
                np.array([0, np.cos(self.theta), np.sin(self.theta)]),
                -state.projected_gravity,
            )
            < self.balance_threshold
        )

    def _check_ballance_upsidedown(self, state) -> bool:
        return (
            np.dot(
                np.array([0, 0, -1]),
                -quat_rotate_inverse(state.accurate_quat, self.gravity_vec),
            )
            < self.orientation_threshold
        )

    def _check_ballance_up(self, state) -> bool:
        return (
            np.dot(
                np.array([0, 0, 1]),
                -quat_rotate_inverse(state.accurate_quat, self.gravity_vec),
            )
            < self.orientation_threshold
        )

    def _check_ballance_auto(self, state) -> bool:
        if self.projected_upward_vec is None:
            raise ValueError("projected_upward_vec required for ballance_auto")
        return (
            np.dot(
                self.projected_upward_vec,
                -quat_rotate_inverse(state.accurate_quat, self.gravity_vec),
            )
            < self.orientation_threshold
        )

    def _check_torso_fall(self, state) -> bool:
        return bool(1 in state.contact_floor_balls or 20 in state.contact_floor_balls)

    def _check_three_feet(self, state) -> bool:
        three_leg = all(x in state.contact_floor_geoms for x in [5, 6, 7]) or all(
            x in state.contact_floor_geoms for x in [9, 10, 11]
        )
        not_moving = np.linalg.norm(state.accurate_vel_world) < 0.1
        return three_leg and not_moving

    def _check_ball_fall(self, state) -> bool:
        return bool(state.contact_floor_balls)

    def _check_body_contact_floor(self, state) -> bool:
        """Check if any of the configured bodies/geoms are in contact with floor.

        Requires set_model() to be called first to map body/geom names to geom IDs.
        """
        if not self._body_contact_geom_ids:
            if self.terminate_on_body_contact or self.terminate_on_geom_contact:
                print(
                    "Warning: body_contact_floor termination strategy is active but "
                    "no geom IDs were found. Did you call set_model()?"
                )
            return False

        # Check if any of the configured geoms are in contact with floor
        contact_geoms = set(state.contact_floor_geoms)
        return bool(self._body_contact_geom_ids & contact_geoms)

    def check_truncated(self, state) -> bool:
        """Check if episode should be truncated (terminated early).

        Args:
            state: Current environment state

        Returns:
            bool: Whether episode should be truncated
        """
        # Check max episode steps
        # print(f"Current step: {self.current_step}, Max steps: {self.max_episode_steps}")
        if (
            self.max_episode_steps is not None
            and self.current_step >= self.max_episode_steps - 1
        ):
            return True
        return False

    def check_upsidedown(self, state) -> bool:
        """Check if robot is upside down.

        Args:
            state: Current environment state

        Returns:
            bool: Whether robot is upside down
        """
        if self.projected_upward_vec is None:
            return None

        accurate_projected_gravity = quat_rotate_inverse(state.quat, self.gravity_vec)
        return (
            np.dot(self.projected_upward_vec, -accurate_projected_gravity)
            < self.orientation_threshold
        )
