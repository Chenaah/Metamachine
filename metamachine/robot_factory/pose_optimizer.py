"""
Pose optimization utilities for robotic systems using JAX-accelerated physics simulation.

This module provides tools for optimizing robot poses through physics simulation,
using JAX for high-performance batch processing and MuJoCo for accurate physics.

Requirements:
    - JAX and JAXlib for accelerated computation
    - MuJoCo-MJX for JAX-accelerated physics simulation
    - Install with: pip install jax jaxlib mujoco-mjx

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

import os
from typing import Any, Optional

import numpy as np
from rich.progress import Progress, TaskID
from scipy.spatial import ConvexHull

# Check for JAX availability
try:
    import jax
    import jax.numpy as jnp
    import mujoco
    from jax._src.typing import Array
    from mujoco import mjx
    from mujoco.mjx._src import types

    HAS_JAX = True
except ImportError as e:
    raise ImportError(
        f"JAX and MuJoCo-MJX are required for pose optimization functionality. "
        f"Install with: pip install 'metamachine[jax]' or pip install jax jaxlib mujoco-mjx\n"
        f"Original error: {e}"
    ) from e

from metamachine import robot_factory
from metamachine.robot_factory.base_factory import BaseRobot
from metamachine.utils.math_utils import quat_rotate_inverse_jax_wxyz
from metamachine.utils.visual_utils import get_joint_pos_addr

# Constants
try:
    GRAVITY_VEC = jnp.array([[0, 0, -1]])
    FALL_THRESHOLD = 0.1
    CONTACT_THRESHOLD = 0.02
    DEFAULT_BATCH_SIZE = 4096
except Exception as e:
    print(f"Error initializing constants: {e}")


def batch_split_almost_evenly(
    lines: jnp.ndarray, points: jnp.ndarray, tolerance: int = 1
) -> jnp.ndarray:
    """
    Checks if each line in a batch approximately splits its corresponding batch of points evenly.

    Parameters:
    - lines: jnp.array of shape (B, 2, 2), where each (2,2) represents a line (two points).
    - points: jnp.array of shape (B, N, 2), where each (N,2) represents N points for a batch.
    - tolerance: Allowed difference in point count between the two sides (default is 1).

    Returns:
    - jnp.array of shape (B,), where each element is True if the line approximately splits the points evenly.
    """

    # Extract line points
    x1, y1 = lines[:, 0, 0], lines[:, 0, 1]  # (B,)
    x2, y2 = lines[:, 1, 0], lines[:, 1, 1]  # (B,)

    # Compute line coefficients: Ax + By + C = 0
    A = y2 - y1  # (B,)
    B = x1 - x2  # (B,)
    C = x2 * y1 - x1 * y2  # (B,)

    # Compute signed distance for each point in batch
    S = A[:, None] * points[..., 0] + B[:, None] * points[..., 1] + C[:, None]  # (B, N)

    # Count points on each side
    N_plus = jnp.sum(S > 0, axis=1)  # (B,)
    N_minus = jnp.sum(S < 0, axis=1)  # (B,)

    # Check if the split is within tolerance
    return jnp.abs(N_plus - N_minus) <= tolerance  # (B,)


def generate_symmetric_list_batch(
    N: int,
    batch_size: int,
    key: Array,
    minval: float = 0,
    maxval: float = 1,
    enforce_mixed_signs: bool = False,
) -> jnp.ndarray:
    """
    Generate a batch of symmetric lists.

    Args:
        N: Length of each list.
        batch_size: Number of lists to generate.
        key: JAX random key.

    Returns:
        A batch of symmetric lists with shape (batch_size, N).
    """
    # Generate random absolute values for each pair
    key, subkey = jax.random.split(key)
    absolute_values = jax.random.uniform(
        subkey, (batch_size, N // 2), minval=minval, maxval=maxval
    )

    # Generate random signs for each pair
    key, subkey = jax.random.split(key)
    if enforce_mixed_signs:
        # Ensure each pair has one positive and one negative value
        signs = jnp.stack(
            [jnp.ones((batch_size, N // 2)), -jnp.ones((batch_size, N // 2))], axis=-1
        )
        # Randomly shuffle the signs within each pair
        key, subkey = jax.random.split(key)
        signs = jax.random.permutation(subkey, signs, axis=-1, independent=True)
    else:
        # Allow any combination of signs (both positive, both negative, or mixed)
        signs = jax.random.choice(
            subkey, jnp.array([-1.0, 1.0]), (batch_size, N // 2, 2)
        )

    # Create symmetric pairs
    symmetric_pairs = absolute_values[..., None] * signs  # Shape: (batch_size, N//2, 2)

    # Reshape to (batch_size, N) for even N
    symmetric_lists = symmetric_pairs.reshape(batch_size, -1)

    # If N is odd, append a 0 to each list
    if N % 2 != 0:
        symmetric_lists = jnp.concatenate(
            [symmetric_lists, jnp.zeros((batch_size, 1))], axis=1
        )

    return symmetric_lists


def is_degenerate(points: np.ndarray) -> bool:
    """Check if the points are degenerate (collinear or duplicate)."""
    # Check if all points are identical
    if np.all(points == points[0]):
        return True
    # Check if points are collinear
    vec1 = points[1] - points[0]
    for i in range(2, len(points)):
        vec2 = points[i] - points[0]
        if np.linalg.norm(np.cross(vec1, vec2)) > 1e-8:
            return False
    return True


def optimize_pose_old(
    morphology: list[int],
    drop_steps: int = 100,
    move_steps: int = 500,
    optimization_type: str = "stablefast",
) -> Any:
    """Old API for pose optimization."""
    # Old API
    if optimization_type in ["longlegs", "bigbase"]:
        move_steps = 0

    factory = robot_factory.get_robot_factory("modular_legs_draft")
    if factory:
        robot = factory.create_robot(morphology=morphology, config=None)
    else:
        raise ValueError("Robot factory not found.")

    return optimize_pose(
        robot, drop_steps, move_steps, optimization_type, enable_progress_bar=False
    )


def optimize_pose(
    robot: BaseRobot,
    drop_steps: int = 150,
    move_steps: int = 250,
    optimization_type: str = "stablefast",
    enable_progress_bar: bool = True,
    spine_assumption: bool = False,
    seed: int = 0,
    log_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Main pose optimization function."""

    xml_string = robot.get_xml_string()

    # Initialize simulation
    mj_model, mjx_model, mjx_data, joint_geom_ids, joint_body_idx, joint_body_ids = (
        setup_simulation(xml_string, seed)
    )

    # Initialize batch poses and run drop simulation
    mjx_data, joint_pos, rand_key = initialize_batch_poses(
        mjx_data, mj_model, seed, spine_assumption
    )

    # Setup progress tracking
    total_steps = drop_steps + move_steps
    progress = Progress()
    task = progress.add_task("[cyan]Optimizing pose...", total=total_steps)
    if enable_progress_bar:
        progress.start()

    try:
        # Run drop simulation
        mjx_data = run_drop_simulation(
            mjx_model, mjx_data, joint_pos, drop_steps, progress, task
        )

        # Extract stable state metrics
        stable_qpos = mjx_data.qpos.copy()
        default_joint_pos = joint_pos.copy()

        stable_metrics = extract_stable_metrics(
            mjx_data, mj_model, joint_geom_ids, joint_body_ids, spine_assumption
        )

        # Run movement phase if needed
        movement_metrics = {}
        if move_steps > 0:
            movement_metrics = run_movement_phase(
                mjx_model,
                mjx_data,
                default_joint_pos,
                move_steps,
                joint_body_idx,
                spine_assumption,
                stable_metrics.get("spine_local_forward"),
                stable_metrics.get("spine_local_upward"),
                stable_metrics.get("projected_upward"),
                progress,
                task,
            )
        else:
            # Initialize default movement metrics for optimization types that don't use movement
            movement_metrics = {
                "avg_vel": jnp.zeros((mjx_data.qpos.shape[0], 2)),
                "avg_speed": jnp.zeros(mjx_data.qpos.shape[0]),
                "avg_projected_vel": jnp.zeros(mjx_data.qpos.shape[0]),
                "fall_down": jnp.zeros(mjx_data.qpos.shape[0], dtype=bool),
            }

        # Calculate final score
        final_score = calculate_optimization_score(
            optimization_type, movement_metrics, stable_metrics, move_steps
        )

        # Extract best result
        result = extract_best_result(
            final_score,
            stable_qpos,
            default_joint_pos,
            movement_metrics,
            stable_metrics,
            spine_assumption,
            log_dir,
        )
    except KeyboardInterrupt:
        print("\n\nPose optimization interrupted by user (Ctrl+C).")
        raise
    finally:
        if enable_progress_bar:
            progress.stop()

    return result


def extract_stable_metrics(
    mjx_data: types.Data,
    mj_model: types.Model,
    joint_geom_ids: list[int],
    joint_body_ids: list[int],
    spine_assumption: bool,
) -> dict[str, Any]:
    """Extract metrics from the stable state after drop simulation."""
    # Joint position metrics
    stable_avg_joint_height = mjx_data.xpos[:, joint_body_ids, 2].mean(axis=1)
    stable_avg_joint_pos = mjx_data.xpos[:, joint_body_ids, :2].mean(axis=1)
    stable_highest_joint_height = mjx_data.xpos[:, joint_body_ids, 2].max(axis=1)
    stable_lowest_joint_height = mjx_data.xpos[:, joint_body_ids, 2].min(axis=1)

    # Contact metrics
    contact_metrics = compute_contact_metrics(mjx_data)

    # Joint contact with floor
    L = jnp.array(joint_geom_ids)
    in_L = jnp.isin(contact_metrics["contact_geom"], L)
    masked_in_L = in_L * contact_metrics["contact_happen_with_floor"][..., None]
    any_in_L = jnp.any(masked_in_L, axis=-1)
    joint_touch_floor = jnp.any(any_in_L, axis=-1)

    # COM distances
    squared_distances = jnp.sum(
        (stable_avg_joint_pos - contact_metrics["avg_contact_pos"]) ** 2, axis=1
    )
    com_distances = jnp.sqrt(squared_distances)

    # Projected vectors
    stable_qpos = mjx_data.qpos
    projected_upward = quat_rotate_inverse_jax_wxyz(
        stable_qpos[:, 3:7], jnp.array([[0, 0, 1]])
    )

    # Spine-specific calculations
    spine_local_upward, spine_local_forward, static_upward_dot = None, None, None
    if spine_assumption:
        spine_local_upward, spine_local_forward = compute_local_vectors(
            stable_qpos[:, 3:7]
        )
        projected_gravity = quat_rotate_inverse_jax_wxyz(
            stable_qpos[:, 3:7], GRAVITY_VEC
        )
        static_upward_dot = jnp.einsum(
            "ij,ij->i", spine_local_upward, -projected_gravity
        )

    # Symmetry check
    spine_lines = mjx_data.geom_xpos[
        :, [mj_model.geom("stick0").id, mj_model.geom("stick1").id], :2
    ]
    other_stick_ids = [mj_model.geom(f"stick{i}").id for i in range(2, 2 * mj_model.nu)]
    leg_module_pos = mjx_data.geom_xpos[:, other_stick_ids, :2]
    sysmetric = batch_split_almost_evenly(spine_lines, leg_module_pos)

    return {
        "stable_avg_joint_height": stable_avg_joint_height,
        "stable_avg_joint_pos": stable_avg_joint_pos,
        "stable_highest_joint_height": stable_highest_joint_height,
        "stable_lowest_joint_height": stable_lowest_joint_height,
        "convex_hull_areas": contact_metrics["convex_hull_areas"],
        "joint_touch_floor": joint_touch_floor,
        "com_distances": com_distances,
        "projected_upward": projected_upward,
        "spine_local_upward": spine_local_upward,
        "spine_local_forward": spine_local_forward,
        "static_upward_dot": static_upward_dot,
        "sysmetric": sysmetric,
    }


def run_movement_phase(
    mjx_model: types.Model,
    mjx_data: types.Data,
    default_joint_pos: jnp.ndarray,
    move_steps: int,
    joint_body_idx: list[int],
    spine_assumption: bool,
    spine_local_forward: Optional[jnp.ndarray],
    spine_local_upward: Optional[jnp.ndarray],
    projected_upward: Optional[jnp.ndarray],
    progress: Progress,
    task: TaskID,
) -> dict[str, jnp.ndarray]:
    """Run the movement phase simulation."""
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    # Initialize tracking variables
    last_com_pos = mjx_data.xpos[:, joint_body_idx, :2].mean(axis=1)
    acc_vel = jnp.zeros(mjx_data.qpos[:, :2].shape)
    acc_projected_vel = jnp.zeros(mjx_data.qpos.shape[0])
    fall_down = jnp.zeros(mjx_data.qpos.shape[0], dtype=bool)

    try:
        for i in range(move_steps):
            # Apply sinusoidal motion
            joint_pos_with_noise = default_joint_pos + jnp.sin(i / 10)
            mjx_data = mjx_data.replace(ctrl=joint_pos_with_noise)
            mjx_data = jit_step(mjx_model, mjx_data)

            # Measure global COM velocity
            com_pos = mjx_data.xpos[:, joint_body_idx, :2].mean(axis=1)
            com_vel = (com_pos - last_com_pos) / mjx_model.opt.timestep
            last_com_pos = com_pos.copy()
            acc_vel += com_vel

            # Measure local spine velocity
            if spine_assumption and spine_local_forward is not None:
                vel_body = mjx_data.qvel[:, 3:6]
                projected_forward_vel = jnp.einsum(
                    "ij,ij->i", spine_local_forward, vel_body
                )
                acc_projected_vel += projected_forward_vel

            # Check for falls
            fall = check_for_falls(
                mjx_data, spine_assumption, spine_local_upward, projected_upward, mjx_model
            )
            fall_down = jnp.logical_or(fall_down, fall)
            progress.advance(task)
    except KeyboardInterrupt:
        print("\nMovement phase interrupted by user.")
        raise

    avg_vel = acc_vel / move_steps
    avg_speed = jnp.linalg.norm(avg_vel, axis=1)
    avg_projected_vel = acc_projected_vel / move_steps

    return {
        "avg_vel": avg_vel,
        "avg_speed": avg_speed,
        "avg_projected_vel": avg_projected_vel,
        "fall_down": fall_down,
    }


def check_for_falls(
    mjx_data: types.Data,
    spine_assumption: bool,
    spine_local_upward: Optional[jnp.ndarray],
    projected_upward: Optional[jnp.ndarray],
    mj_model: types.Model,
) -> jnp.ndarray:
    """Check if robots have fallen down."""
    quat = mjx_data.qpos[:, 3:7]
    projected_gravity = quat_rotate_inverse_jax_wxyz(quat, GRAVITY_VEC)

    if not spine_assumption:
        dot_results = jnp.einsum("ij,ij->i", projected_upward, -projected_gravity)
        return dot_results < FALL_THRESHOLD
    else:
        dot_results = jnp.einsum("ij,ij->i", spine_local_upward, -projected_gravity)
        wall_fall = dot_results < FALL_THRESHOLD

        # Check contact with floor
        contact_geom = mjx_data.contact.geom
        dist = mjx_data.contact.dist
        contact_floor = jnp.any(contact_geom == 0, axis=2)
        contact_happen = dist < CONTACT_THRESHOLD
        contact_happen_with_floor = contact_floor & contact_happen
        torso_geom_ids = [mj_model.geom("left0").id, mj_model.geom("right0").id]
        L = jnp.array(torso_geom_ids)
        in_L = jnp.isin(contact_geom, L)
        masked_in_L = in_L * contact_happen_with_floor[..., None]
        any_in_L = jnp.any(masked_in_L, axis=-1)
        joint_touch_floor = jnp.any(any_in_L, axis=-1)
        return jnp.logical_or(joint_touch_floor, wall_fall)


def extract_best_result(
    final_score: jnp.ndarray,
    stable_qpos: jnp.ndarray,
    default_joint_pos: jnp.ndarray,
    movement_metrics: dict[str, jnp.ndarray],
    stable_metrics: dict[str, Any],
    spine_assumption: bool,
    log_dir: Optional[str],
) -> dict[str, Any]:
    """Extract the best result from the optimization."""
    idx = jnp.argmax(final_score)
    best_stable_qpos = stable_qpos[idx]

    init_pos = [0, 0, best_stable_qpos[2].item() + 0.01]
    init_quat = best_stable_qpos[3:7].tolist()
    init_joint = default_joint_pos[idx].tolist()

    forward_vec = jnp.append(movement_metrics["avg_vel"][idx], 0)
    forward_vec = forward_vec / jnp.linalg.norm(forward_vec)

    info = {"forward_vec": forward_vec.tolist(), "score": final_score[idx].item()}

    if not spine_assumption:
        info["projected_upward"] = stable_metrics["projected_upward"][idx].tolist()
        projected_forward = quat_rotate_inverse_jax_wxyz(
            stable_qpos[:, 3:7], jnp.expand_dims(forward_vec, 0)
        )[idx]
        info["projected_forward"] = projected_forward.tolist()
    else:
        info["projected_forward"] = stable_metrics["spine_local_forward"][idx].tolist()
        info["projected_upward"] = stable_metrics["spine_local_upward"][idx].tolist()

    if log_dir is not None:
        np.savez_compressed(
            os.path.join(log_dir, "poses.npz"),
            stable_qpos=stable_qpos,
            default_joint_pos=default_joint_pos,
            score=final_score,
            best_stable_qpos=best_stable_qpos,
            best_init_pos=init_pos,
            best_init_quat=init_quat,
            best_init_joint=init_joint,
            best_forward_vec=forward_vec,
            best_projected_upward=info["projected_upward"],
            best_projected_forward=info["projected_forward"],
        )

    info["init_pos"] = init_pos
    info["init_quat"] = init_quat
    info["default_dof_pos"] = init_joint

    return info


def quaternion_to_euler(
    quaternion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    Quaternion should be in the form [w, x, y, z].
    """
    w, x, y, z = quaternion
    # Convert quaternion to rotation matrix components
    r00 = 1 - 2 * (y * y + z * z)
    2 * (x * y - z * w)
    2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    1 - 2 * (x * x + z * z)
    2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    # Extract Euler angles
    roll = jnp.arctan2(r21, r22)
    pitch = jnp.arcsin(-r20)
    yaw = jnp.arctan2(r10, r00)

    return roll, pitch, yaw


def compute_local_vectors(quaternions: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute local upward and forward vectors for a batch of quaternions."""
    batched_quaternion_to_euler = jax.vmap(quaternion_to_euler, in_axes=(0,))
    euler_angles = batched_quaternion_to_euler(quaternions)
    rolls = euler_angles[0]  # Extract roll angles (shape: [N,])

    # Compute local_upward and local_forward for each roll
    sin_roll = jnp.sin(rolls)
    cos_roll = jnp.cos(rolls)

    local_upward = jnp.stack([jnp.zeros_like(rolls), sin_roll, cos_roll], axis=-1)
    local_forward = jnp.stack([jnp.zeros_like(rolls), cos_roll, -sin_roll], axis=-1)

    return local_upward, local_forward


def compute_convex_hull_areas(
    contact_pos: jnp.ndarray, contact_happen_with_floor: jnp.ndarray
) -> jnp.ndarray:
    """Calculate the convex hull area for each batch."""
    areas = []
    for cp, c in zip(contact_pos, contact_happen_with_floor):
        floor_points = cp[c]
        floor_points = np.asarray(floor_points)[:, :2]
        if len(floor_points) < 3 or is_degenerate(floor_points):
            area = 0.0
        else:
            area = ConvexHull(floor_points).volume
        areas.append(area)
    return jnp.array(areas)


def compute_contact_metrics(mjx_data: types.Data) -> dict[str, jnp.ndarray]:
    """Compute contact-related metrics."""
    contact_geom = mjx_data.contact.geom
    dist = mjx_data.contact.dist
    contact_pos = mjx_data.contact.pos
    contact_floor = jnp.any(contact_geom == 0, axis=2)
    contact_happen = dist < CONTACT_THRESHOLD
    contact_happen_with_floor = contact_floor & contact_happen

    contact_happen_with_floor_bc = jnp.broadcast_to(
        contact_happen_with_floor[:, :, None], contact_pos.shape
    )
    floor_contact_pos = jnp.where(contact_happen_with_floor_bc, contact_pos, 0.0)
    contact_pos_sum = jnp.sum(floor_contact_pos, axis=1)
    count_contact = jnp.sum(contact_happen_with_floor, axis=1, keepdims=True)
    count_contact = jnp.maximum(count_contact, 1)  # To avoid division by zero
    avg_contact_pos = (contact_pos_sum / count_contact)[:, :2]

    convex_hull_areas = compute_convex_hull_areas(
        contact_pos, contact_happen_with_floor
    )

    return {
        "contact_geom": contact_geom,
        "contact_happen_with_floor": contact_happen_with_floor,
        "avg_contact_pos": avg_contact_pos,
        "convex_hull_areas": convex_hull_areas,
    }


def setup_simulation(xml: str, seed: int) -> tuple:
    """Initialize the simulation environment."""
    if os.path.isfile(xml):
        mj_model = mujoco.MjModel.from_xml_path(xml)
    else:
        mj_model = mujoco.MjModel.from_xml_string(xml)

    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    # Setup geometric IDs and indices
    joint_geom_ids = [mj_model.geom(f"left{i}").id for i in range(mj_model.nu)] + [
        mj_model.geom(f"right{i}").id for i in range(mj_model.nu)
    ]
    joint_unique_geom_idx = [mj_model.geom(f"left{i}").id for i in range(mj_model.nu)]
    joint_body_idx = [mj_model.geom(i).bodyid.item() for i in joint_unique_geom_idx]
    joint_body_ids = [mj_model.body(f"l{i}").id for i in range(mj_model.nu)]

    return mj_model, mjx_model, mjx_data, joint_geom_ids, joint_body_idx, joint_body_ids


def initialize_batch_poses(
    mjx_data: types.Data, mj_model: mujoco.MjModel, seed: int, spine_assumption: bool
) -> tuple[types.Data, jnp.ndarray, Array]:
    """Initialize random poses for the batch."""

    def set_random_qpos(rng: Array) -> types.Data:
        fixed_pos = jnp.array([0.0, 0.0, 0.4])
        quaternion = jax.random.uniform(rng, (4,), minval=-1.0, maxval=1.0)
        norm = jnp.linalg.norm(quaternion)
        quaternion = quaternion / norm
        qpos_len = mjx_data.qpos.shape[0]
        remaining_qpos = jax.random.uniform(
            rng, (qpos_len - 7,), minval=-jnp.pi, maxval=jnp.pi
        )
        new_qpos = jnp.concatenate([fixed_pos, quaternion, remaining_qpos])
        return mjx_data.replace(qpos=new_qpos)

    rand_key = jax.random.PRNGKey(seed)
    print(f"Seed: {seed}")
    rngs = jax.random.split(rand_key, DEFAULT_BATCH_SIZE)
    mjx_data = jax.vmap(set_random_qpos, in_axes=0)(rngs)

    if spine_assumption:
        spine_idx = get_joint_pos_addr(mj_model)[0]
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[:, spine_idx].set(0))
        n_envs, qpos_len = mjx_data.qpos.shape
        remaining_joint_idx = get_joint_pos_addr(mj_model)[1:]
        remaining_joint_pos = generate_symmetric_list_batch(
            qpos_len - 7 - 1,
            n_envs,
            rand_key,
            minval=0,
            maxval=1,
            enforce_mixed_signs=False,
        )
        mjx_data = mjx_data.replace(
            qpos=mjx_data.qpos.at[:, remaining_joint_idx].set(remaining_joint_pos)
        )

    joint_pos = mjx_data.qpos[:, get_joint_pos_addr(mj_model)]
    return mjx_data, joint_pos, rand_key


def run_drop_simulation(
    mjx_model: types.Model,
    mjx_data: types.Data,
    joint_pos: jnp.ndarray,
    drop_steps: int,
    progress: Progress,
    task: TaskID,
) -> types.Data:
    """Run the drop simulation phase."""
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    try:
        for _i in range(drop_steps):
            mjx_data = mjx_data.replace(ctrl=joint_pos)
            mjx_data = jit_step(mjx_model, mjx_data)
            progress.advance(task)
    except KeyboardInterrupt:
        print("\nDrop simulation interrupted by user.")
        raise

    return mjx_data


def calculate_optimization_score(
    optimization_type: str,
    metrics: dict[str, jnp.ndarray],
    stable_metrics: dict[str, jnp.ndarray],
    move_steps: int,
) -> jnp.ndarray:
    """Calculate the final optimization score based on the optimization type."""
    score_functions = {
        "stablefast": lambda: jnp.where(
            metrics["fall_down"], metrics["avg_speed"] - 100, metrics["avg_speed"]
        ),
        "stablefastair": lambda: jnp.where(
            jnp.logical_or(metrics["fall_down"], stable_metrics["joint_touch_floor"]),
            metrics["avg_speed"] - 100,
            metrics["avg_speed"],
        ),
        "longlegs": lambda: _calculate_longlegs_score(stable_metrics, move_steps),
        "bigbase": lambda: _calculate_bigbase_score(stable_metrics, move_steps),
        "fastbigbase": lambda: _calculate_fastbigbase_score(
            stable_metrics, metrics, move_steps
        ),
        "stablefastspine": lambda: jnp.where(
            metrics["fall_down"],
            metrics["avg_projected_vel"]
            + 0.2 * stable_metrics["static_upward_dot"]
            - 100,
            metrics["avg_projected_vel"] + 0.2 * stable_metrics["static_upward_dot"],
        ),
        "stablespine": lambda: jnp.where(
            metrics["fall_down"],
            stable_metrics["convex_hull_areas"] - 100,
            stable_metrics["convex_hull_areas"],
        ),
        "sysmetricstablespine": lambda: _calculate_symmetric_stable_spine_score(
            stable_metrics, metrics
        ),
        "sysmetricstablespineair": lambda: _calculate_symmetric_stable_spine_air_score(
            stable_metrics, metrics
        ),
    }

    if optimization_type in score_functions:
        return score_functions[optimization_type]()
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")


def _calculate_longlegs_score(
    stable_metrics: dict[str, jnp.ndarray], move_steps: int
) -> jnp.ndarray:
    """Calculate score for longlegs optimization."""
    if move_steps != 0:
        print("Warning: move_steps should be 0 for longlegs optimization")

    final_score = stable_metrics["stable_avg_joint_height"]
    final_score = jnp.where(stable_metrics["com_distances"] < 0.1, final_score, -999)
    final_score = jnp.where(
        jnp.sqrt(stable_metrics["convex_hull_areas"])
        < stable_metrics["stable_highest_joint_height"],
        final_score,
        -999,
    )
    return final_score


def _calculate_bigbase_score(
    stable_metrics: dict[str, jnp.ndarray], move_steps: int
) -> jnp.ndarray:
    """Calculate score for bigbase optimization."""
    if move_steps != 0:
        print("Warning: move_steps should be 0 for bigbase optimization")

    print("Big Base Optimization!")
    final_score = (
        stable_metrics["convex_hull_areas"]
        * stable_metrics["stable_lowest_joint_height"]
    )
    final_score = jnp.where(
        jnp.sqrt(stable_metrics["convex_hull_areas"])
        > stable_metrics["stable_highest_joint_height"],
        final_score,
        final_score - 999,
    )
    final_score = jnp.where(
        stable_metrics["joint_touch_floor"], final_score - 999, final_score
    )
    return final_score


def _calculate_fastbigbase_score(
    stable_metrics: dict[str, jnp.ndarray],
    metrics: dict[str, jnp.ndarray],
    move_steps: int,
) -> jnp.ndarray:
    """Calculate score for fastbigbase optimization."""
    if move_steps == 0:
        print("Warning: move_steps should be non-0 for fastbigbase optimization")

    print("Fast Big Base Optimization!")
    final_score = (
        stable_metrics["convex_hull_areas"]
        * stable_metrics["stable_lowest_joint_height"]
        + metrics["avg_speed"]
    )
    final_score = jnp.where(
        jnp.sqrt(stable_metrics["convex_hull_areas"])
        > stable_metrics["stable_highest_joint_height"],
        final_score,
        final_score - 100,
    )
    final_score = jnp.where(
        stable_metrics["joint_touch_floor"], final_score - 100, final_score
    )
    final_score = jnp.where(metrics["fall_down"], final_score - 100, final_score)
    return final_score


def _calculate_symmetric_stable_spine_score(
    stable_metrics: dict[str, jnp.ndarray], metrics: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Calculate score for symmetric stable spine optimization."""
    final_score = stable_metrics["convex_hull_areas"]
    final_score = jnp.where(metrics["fall_down"], final_score - 100, final_score)
    final_score = jnp.where(stable_metrics["sysmetric"], final_score, final_score - 100)
    return final_score


def _calculate_symmetric_stable_spine_air_score(
    stable_metrics: dict[str, jnp.ndarray], metrics: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Calculate score for symmetric stable spine air optimization."""
    final_score = stable_metrics["convex_hull_areas"]
    final_score = jnp.where(metrics["fall_down"], final_score - 100, final_score)
    final_score = jnp.where(
        stable_metrics["joint_touch_floor"], final_score - 100, final_score
    )
    final_score = jnp.where(stable_metrics["sysmetric"], final_score, final_score - 100)
    return final_score
