"""
MuJoCo runtime utilities for modular robot analysis.

This module provides utility functions for analyzing MuJoCo models at runtime,
particularly for finding clusters of connected bodies via weld equality constraints.

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

from collections import defaultdict
from typing import Optional

import mujoco
import numpy as np


def find_parent_torso(model: mujoco.MjModel, body_id: int) -> Optional[int]:
    """
    Find the parent torso body (body with freejoint) for a given body.
    
    Traverses up the body hierarchy to find the first body that has a freejoint,
    which is considered a "torso" in modular robots.
    
    Args:
        model: MuJoCo model
        body_id: ID of the body to start from
        
    Returns:
        Body ID of the parent torso, or None if no torso found
    """
    current_id = body_id
    
    while current_id > 0:  # 0 is world body
        # Check if this body has a freejoint
        for joint_id in range(model.njnt):
            if model.jnt_bodyid[joint_id] == current_id:
                if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    return current_id
        
        # Move to parent body
        current_id = model.body_parentid[current_id]
    
    return None


def get_all_weld_clusters(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[int]]:
    """
    Find all clusters of connected torso bodies (bodies with freejoints) via weld equality constraints.
    
    This function:
    1. Finds all bodies with freejoints (torso bodies)
    2. Builds a graph of connections based on active weld constraints
    3. Returns all connected components (clusters)
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        List of clusters, where each cluster is a list of torso body IDs
    """
    # Step 1: Find all torso bodies (bodies with freejoints)
    all_torsos = []
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name and body_name.startswith('torso'):
            # Verify it has a freejoint
            for joint_id in range(model.njnt):
                if model.jnt_bodyid[joint_id] == body_id:
                    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                        all_torsos.append(body_id)
                        break
    
    # Step 2: Build graph of connected torso bodies
    graph = defaultdict(set)
    
    # Initialize graph with all torsos (even isolated ones)
    for torso_id in all_torsos:
        graph[torso_id] = set()

    for i in range(model.eq_type.shape[0]):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD and data.eq_active[i]:
            site1_id = model.eq_obj1id[i]
            site2_id = model.eq_obj2id[i]
            
            # Get the bodies that these sites belong to
            body1_id = model.site_bodyid[site1_id]
            body2_id = model.site_bodyid[site2_id]
            
            # Find the parent torso bodies (bodies with freejoints)
            torso1_id = find_parent_torso(model, body1_id)
            torso2_id = find_parent_torso(model, body2_id)
            
            if torso1_id is not None and torso2_id is not None and torso1_id != torso2_id:
                # Add bidirectional connection between torsos
                graph[torso1_id].add(torso2_id)
                graph[torso2_id].add(torso1_id)

    # Step 3: Find connected components using DFS
    visited = set()
    components = []

    def dfs(node: int, comp: list):
        visited.add(node)
        comp.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, comp)

    for node in graph:
        if node not in visited:
            comp = []
            dfs(node, comp)
            components.append(comp)

    return components


def get_largest_weld_cluster_average_pos(
    model: mujoco.MjModel, 
    data: mujoco.MjData
) -> tuple[Optional[list[int]], Optional[np.ndarray], Optional[list[str]]]:
    """
    Find the largest cluster of connected torso bodies (bodies with freejoints) 
    via weld equality constraints and compute the average position of this cluster.
    
    This is useful for tracking the center of mass of a modular robot where
    multiple body segments are welded together.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        tuple: (largest_cluster, avg_pos, body_names) where:
               - largest_cluster: list of torso body IDs in the largest cluster
               - avg_pos: numpy array of the average position (x, y, z)
               - body_names: list of body names for the IDs in largest_cluster
               Returns (None, None, None) if no clusters found.
    """
    # Get all clusters
    all_clusters = get_all_weld_clusters(model, data)
    
    if not all_clusters:
        return None, None, None  # No clusters found

    # Find the largest cluster
    largest_cluster = max(all_clusters, key=len)

    # Compute average position of torso bodies in the largest cluster
    torso_positions = np.array([data.xpos[body_id] for body_id in largest_cluster])
    avg_pos = np.mean(torso_positions, axis=0)

    # Get body names for each body_id in largest_cluster
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) 
        for body_id in largest_cluster
    ]

    return largest_cluster, avg_pos, body_names


def get_weld_cluster_center_of_mass(
    model: mujoco.MjModel, 
    data: mujoco.MjData
) -> Optional[np.ndarray]:
    """
    Get the mass-weighted center of mass of the largest weld cluster.
    
    Unlike get_largest_weld_cluster_average_pos which computes simple position average,
    this function computes the proper center of mass weighted by body masses.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        
    Returns:
        numpy array of the center of mass position (x, y, z), or None if no clusters
    """
    all_clusters = get_all_weld_clusters(model, data)
    
    if not all_clusters:
        return None

    largest_cluster = max(all_clusters, key=len)
    
    total_mass = 0.0
    weighted_pos = np.zeros(3)
    
    for body_id in largest_cluster:
        mass = model.body_mass[body_id]
        pos = data.xpos[body_id]
        weighted_pos += mass * pos
        total_mass += mass
    
    if total_mass > 0:
        return weighted_pos / total_mass
    else:
        # Fallback to simple average if all masses are zero
        return np.mean([data.xpos[body_id] for body_id in largest_cluster], axis=0)



