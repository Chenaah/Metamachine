#!/usr/bin/env python3
"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

MuJoCo XML Hierarchy Transformer - Final Version

This module provides functionality to change the parent-child hierarchy of bodies 
in a MuJoCo XML file while maintaining physical equivalence.

Usage Examples:
    # Transform specific case (passive2 becomes parent of torso1)
    transform_torso1_passive2_hierarchy(input_file, output_file)
    
    # General case
    change_to_root_with_new_parent(input_file, "child_body", "new_parent_body", output_file)
"""

import pdb
import xml.etree.ElementTree as ET
import numpy as np


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_conjugate(q):
    """Return the conjugate of a quaternion (w, x, y, z format)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def parse_pos_quat(element):
    """Parse position and quaternion from XML element."""
    pos_str = element.get('pos', '0 0 0')
    quat_str = element.get('quat', '1 0 0 0')
    
    pos = np.array([float(x) for x in pos_str.split()])
    quat = np.array([float(x) for x in quat_str.split()])
    
    return pos, quat


def set_pos_quat(element, pos, quat):
    """Set position and quaternion attributes in XML element."""
    element.set('pos', ' '.join([str(x) for x in pos]))
    element.set('quat', ' '.join([str(x) for x in quat]))


def transform_pose(pos, quat, parent_pos, parent_quat):
    """Transform pose from parent coordinate frame to world frame."""
    R_parent = quaternion_to_rotation_matrix(parent_quat)
    world_pos = parent_pos + R_parent @ pos
    world_quat = quaternion_multiply(parent_quat, quat)
    return world_pos, world_quat


def inverse_transform_pose(world_pos, world_quat, new_parent_pos, new_parent_quat):
    """Transform pose from world frame to new parent coordinate frame."""
    R_new_parent_inv = quaternion_to_rotation_matrix(quaternion_conjugate(new_parent_quat))
    relative_pos = R_new_parent_inv @ (world_pos - new_parent_pos)
    relative_quat = quaternion_multiply(quaternion_conjugate(new_parent_quat), world_quat)
    return relative_pos, relative_quat


def find_body_by_name(root, body_name):
    """Find a body element by name in the XML tree."""
    for body in root.iter('body'):
        if body.get('name') == body_name:
            return body
    return None


def find_parent_of_body(root, target_body):
    """Find the parent element of a given body."""
    for elem in root.iter():
        for child in elem:
            if child is target_body:
                return elem
    return None


def get_body_chain_to_root(root, body):
    """Get the chain of bodies from given body to root (worldbody)."""
    chain = []
    current = body
    
    while current is not None and current.tag == 'body':
        chain.append(current)
        current = find_parent_of_body(root, current)
        
    return chain


def compute_world_pose(body_chain):
    """Compute world pose from a chain of bodies."""
    world_pos = np.array([0.0, 0.0, 0.0])
    world_quat = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Start from the root and work down (reverse order)
    for body in reversed(body_chain):
        pos, quat = parse_pos_quat(body)
        world_pos, world_quat = transform_pose(world_pos, world_quat, pos, quat)
    
    return world_pos, world_quat


def change_to_root_with_new_parent(xml_file_path, target_body_name, new_parent_body_name, output_file_path=None, verbose=False):
    """
    Change hierarchy so that new_parent becomes the root with freejoint, and target becomes its child.
    
    This function handles the complex case where we want to make a descendant body become 
    the parent of its current ancestor while maintaining physical equivalence.
    
    Args:
        xml_file_path: Path to input MuJoCo XML file
        target_body_name: Name of the body that currently has the freejoint
        new_parent_body_name: Name of the body that should become the new parent
        output_file_path: Path for output file (optional)
        verbose: Print detailed transformation information
    
    Returns:
        Path to the transformed XML file
    
    The transformation maintains physical equivalence by:
    1. Computing world poses of all relevant bodies
    2. Moving the freejoint to the new parent
    3. Updating all relative poses to maintain the same world poses
    4. Restructuring the XML hierarchy
    """
    # Parse XML
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Find target and new parent bodies
    target_body = find_body_by_name(root, target_body_name)
    new_parent_body = find_body_by_name(root, new_parent_body_name)
    
    if target_body is None:
        raise ValueError(f"Target body '{target_body_name}' not found")
    if new_parent_body is None:
        raise ValueError(f"New parent body '{new_parent_body_name}' not found")
    
    # Get worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No worldbody found in XML")
    
    if verbose:
        print(f"Transforming hierarchy: making '{new_parent_body_name}' the parent of '{target_body_name}'")
    
    # Step 1: Compute world poses for all relevant bodies
    target_chain = get_body_chain_to_root(root, target_body)
    target_world_pos, target_world_quat = compute_world_pose(target_chain)
    
    new_parent_chain = get_body_chain_to_root(root, new_parent_body)
    new_parent_world_pos, new_parent_world_quat = compute_world_pose(new_parent_chain)
    
    if verbose:
        print(f"Target world pose: pos={target_world_pos}, quat={target_world_quat}")
        print(f"New parent world pose: pos={new_parent_world_pos}, quat={new_parent_world_quat}")
    
    # Step 2: Find and move the freejoint
    freejoint = target_body.find('freejoint')
    if freejoint is not None:
        if verbose:
            print("Moving freejoint from target to new parent")
        target_body.remove(freejoint)
        # Insert freejoint as first child of new_parent
        new_parent_body.insert(0, freejoint)
    
    # Step 3: Remove new_parent from its current location
    new_parent_current_parent = find_parent_of_body(root, new_parent_body)
    if new_parent_current_parent is not None:
        if verbose:
            print(f"Removing new parent from its current parent: {new_parent_current_parent.get('name', 'worldbody')}")
        new_parent_current_parent.remove(new_parent_body)
    
    # Step 4: Set new_parent's world pose (it becomes the root)
    set_pos_quat(new_parent_body, new_parent_world_pos, new_parent_world_quat)
    
    # Step 5: Calculate target's pose relative to new_parent
    target_rel_pos, target_rel_quat = inverse_transform_pose(
        target_world_pos, target_world_quat,
        new_parent_world_pos, new_parent_world_quat
    )
    
    if verbose:
        print(f"Target relative pose: pos={target_rel_pos}, quat={target_rel_quat}")
    
    # Step 6: Update target body's pose and remove from current parent
    set_pos_quat(target_body, target_rel_pos, target_rel_quat)
    target_current_parent = find_parent_of_body(root, target_body)
    if target_current_parent is not None:
        if verbose:
            print(f"Removing target from its current parent: {target_current_parent.get('name', 'worldbody')}")
        target_current_parent.remove(target_body)
    
    # Step 7: Reconstruct hierarchy
    # Check if new_parent is a descendant of target (would create circular reference)
    def is_descendant(ancestor, potential_descendant, root):
        """Check if potential_descendant is a descendant of ancestor"""
        for elem in ancestor.iter():
            if elem is potential_descendant:
                return True
        return False
    
    if is_descendant(target_body, new_parent_body, root):
        print("WARNING: Detected potential circular reference - aborting transformation")
        print(f"'{new_parent_body_name}' is currently a descendant of '{target_body_name}'")
        pdb.set_trace()
        return None
    
    # Add new_parent to worldbody (it becomes the root)
    worldbody.append(new_parent_body)
    
    # Add target as child of new_parent
    new_parent_body.append(target_body)
    
    if verbose:
        print("Hierarchy reconstruction complete")
    
    # Validate tree integrity before writing
    def check_tree_integrity(element, visited=None, depth=0, max_depth=100):
        """Check for circular references and excessive depth in XML tree."""
        if visited is None:
            visited = set()
        
        element_id = id(element)
        if element_id in visited:
            print(f"CIRCULAR REFERENCE detected at depth {depth}: {element.tag} {element.get('name', 'unnamed')}")
            return False
        
        if depth > max_depth:
            print(f"Excessive depth detected at {depth}: {element.tag} {element.get('name', 'unnamed')}")
            return False
        
        visited.add(element_id)
        
        for child in element:
            if not check_tree_integrity(child, visited.copy(), depth + 1, max_depth):
                return False
        
        return True
    
    def validate_hierarchy_structure(root):
        """Validate the specific hierarchy requirements."""
        # Check that both bodies still exist
        target_found = find_body_by_name(root, target_body_name)
        new_parent_found = find_body_by_name(root, new_parent_body_name)
        
        if target_found is None:
            print(f"ERROR: Target body '{target_body_name}' not found in tree")
            return False
        if new_parent_found is None:
            print(f"ERROR: New parent body '{new_parent_body_name}' not found in tree")
            return False
        
        # Check that new_parent is in worldbody
        worldbody = root.find('worldbody')
        if new_parent_found not in worldbody:
            print(f"ERROR: New parent '{new_parent_body_name}' is not a direct child of worldbody")
            return False
        
        # Check that target is a child of new_parent
        if target_found not in new_parent_found:
            print(f"ERROR: Target '{target_body_name}' is not a child of new parent '{new_parent_body_name}'")
            return False
        
        # Check that freejoint is in new_parent
        freejoint_in_new_parent = new_parent_found.find('freejoint')
        if freejoint_in_new_parent is None:
            print(f"ERROR: No freejoint found in new parent '{new_parent_body_name}'")
            return False
        
        # Check that target doesn't have freejoint
        freejoint_in_target = target_found.find('freejoint')
        if freejoint_in_target is not None:
            print(f"ERROR: Target '{target_body_name}' still has freejoint")
            return False
        
        return True
    
    if verbose:
        print("Validating tree integrity...")
    
    # Check for circular references and corruption
    if not check_tree_integrity(root):
        print("ERROR: Tree integrity check failed - corrupted XML structure detected!")
        return None
    
    # Check hierarchy structure
    if not validate_hierarchy_structure(root):
        print("ERROR: Hierarchy validation failed!")
        return None
    
    if verbose:
        print("Tree validation passed successfully")
    
    # Set output file path
    if output_file_path is None:
        base_name = xml_file_path.rsplit('.xml', 1)[0]
        output_file_path = f"{base_name}_reordered.xml"
    
    # Test serialization with a small sample first
    try:
        if verbose:
            print("Testing XML serialization...")
        test_string = ET.tostring(root, encoding='utf-8')[:1000]  # Test first 1000 chars
        if verbose:
            print("Serialization test passed")
    except Exception as e:
        print(f"ERROR: Serialization test failed: {e}")
        return None
    
    # Write modified XML
    try:
        tree.write(output_file_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        print(f"ERROR: Failed to write XML file: {e}")
        return None
    
    
    if verbose:
        print(f"Modified XML saved to: {output_file_path}")
    
    return output_file_path


def transform_torso1_passive2_hierarchy(xml_file_path, output_file_path=None, verbose=False):
    """
    Transform the specific case: make passive2 the parent of torso1 with the freejoint.
    
    This will:
    1. Move the freejoint from torso1 to passive2
    2. Make passive2 a root body in worldbody  
    3. Make torso1 a child of passive2
    4. Update all poses to maintain physical equivalence
    
    Args:
        xml_file_path: Path to input XML file
        output_file_path: Path for output file (optional)
        verbose: Print transformation details
    
    Returns:
        Path to the transformed XML file
    """
    return change_to_root_with_new_parent(xml_file_path, "torso1", "passive2", output_file_path, verbose)


# Validation function
def validate_transformation(original_file, transformed_file):
    """
    Validate that the transformation preserves all important elements.
    
    Args:
        original_file: Path to original XML file
        transformed_file: Path to transformed XML file
    
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Parse both files
    orig_tree = ET.parse(original_file)
    trans_tree = ET.parse(transformed_file)
    
    # Check that all bodies still exist
    orig_bodies = {body.get('name') for body in orig_tree.iter('body')}
    trans_bodies = {body.get('name') for body in trans_tree.iter('body')}
    results['bodies_preserved'] = orig_bodies == trans_bodies
    
    # Check that actuators are preserved
    orig_actuators = len(list(orig_tree.iter('position')))
    trans_actuators = len(list(trans_tree.iter('position')))
    results['actuators_preserved'] = orig_actuators == trans_actuators
    
    # Check that sensors are preserved
    orig_sensor_elem = orig_tree.find('sensor')
    trans_sensor_elem = trans_tree.find('sensor')
    orig_sensors = len(list(orig_sensor_elem)) if orig_sensor_elem is not None else 0
    trans_sensors = len(list(trans_sensor_elem)) if trans_sensor_elem is not None else 0
    results['sensors_preserved'] = orig_sensors == trans_sensors
    
    # Check that exactly one freejoint exists in each
    orig_freejoints = len(list(orig_tree.iter('freejoint')))
    trans_freejoints = len(list(trans_tree.iter('freejoint')))
    results['freejoint_count_correct'] = orig_freejoints == trans_freejoints == 1
    
    return results


