#!/usr/bin/env python3
"""
Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Modular Robot Hierarchy Restructuring Script

This script:
1. Uses analyze_xml_structure to get a reduction plan
2. Transforms child modules so that bodies with docking sites become roots
3. Positions child modules as children of the root module using get_B_in_A
4. Creates a unified hierarchy that eliminates the need for weld constraints

"""

import pdb
import xml.etree.ElementTree as ET
import numpy as np
import copy
from pathlib import Path
import sys
import os


# Import required modules
from metamachine.robot_factory.core.mujoco_hierarchy_transformer import change_to_root_with_new_parent
from metamachine.utils.docking_utils import get_B_in_A
from metamachine.utils.xml_graph_utils import (
    create_parent_map, find_module_bodies, map_sites_to_modules,
    build_connection_graph, select_root_module, build_spanning_tree,
    print_tree_structure, analyze_freejoint_reduction
)


def parse_site_pose(site_element):
    """Parse position and quaternion from a site element."""
    pos_str = site_element.get('pos', '0 0 0')
    quat_str = site_element.get('quat', '1 0 0 0')
    
    pos = np.array([float(x) for x in pos_str.split()])
    quat = np.array([float(x) for x in quat_str.split()])
    
    return pos, quat


def find_site_by_name(root, site_name):
    """Find a site element by name in the XML tree."""
    for site in root.iter('site'):
        if site.get('name') == site_name:
            return site
    return None


def find_body_containing_site(root, site_name):
    """Find the body that contains a specific site."""
    parent_map = create_parent_map(root)
    site_element = find_site_by_name(root, site_name)
    
    if site_element is None:
        return None
    
    # Walk up the tree to find the containing body
    current = site_element
    while current is not None:
        current = parent_map.get(current)
        if current is not None and current.tag == 'body':
            return current
    
    return None


def transform_module_hierarchy(xml_path, module_name, docking_site_name, output_path):
    """
    Transform a module so that the body containing the docking site becomes the root.
    
    Args:
        xml_path: Path to the module XML file
        module_name: Name of the module to transform
        docking_site_name: Name of the docking site
        output_path: Path for the transformed XML
    
    Returns:
        Path to the transformed XML file
    """
    print(f"Transforming module {module_name} to make docking site {docking_site_name} the root...")
    
    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find the body containing the docking site
    docking_body = find_body_containing_site(root, docking_site_name)
    if docking_body is None:
        raise ValueError(f"Could not find body containing site {docking_site_name}")
    
    docking_body_name = docking_body.get('name')
    print(f"  Docking site {docking_site_name} is in body {docking_body_name}")
    
    if docking_body_name == module_name:
        print(f"  Docking body is already the root of module {module_name}, no transformation needed.")
        return xml_path

    # Transform the hierarchy to make the docking body the root
    try:
        transformed_path = change_to_root_with_new_parent(
            xml_path, module_name, docking_body_name, output_path, verbose=True
        )
        print(f"  ✓ Transformed module saved to {transformed_path}")
        return transformed_path
    except Exception as e:
        print(f"  ✗ Failed to transform module {module_name}: {e}")
        return None


def get_reduction_plan(xml_path):
    """
    Analyze XML structure and return the reduction plan.
    
    Returns:
        dict containing modules, connections, spanning_tree, root_module, etc.
    """
    print("=" * 60)
    print("ANALYZING XML STRUCTURE FOR REDUCTION PLAN")
    print("=" * 60)
    
    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create parent mapping
    parent_map = create_parent_map(root)
    
    # Find all modules
    modules = find_module_bodies(root)
    print(f"Found {len(modules)} modules: {list(modules.keys())}")
    
    # Map sites to modules
    site_to_module = map_sites_to_modules(root, modules)
    print(f"Found {len(site_to_module)} docking sites")
    
    # Build connection graph
    connections = build_connection_graph(root, modules, site_to_module)
    
    # Select root module
    root_module = select_root_module(connections)
    print(f"Selected root module: {root_module}")
    
    # Build spanning tree
    spanning_tree, tree_structure = build_spanning_tree(connections, root_module)
    print(f"Built spanning tree with {len(spanning_tree)} edges")
    
    # Analyze freejoint reduction
    analysis = analyze_freejoint_reduction(modules, spanning_tree)
    
    return {
        'tree': tree,
        'root': root,
        'modules': modules,
        'site_to_module': site_to_module,
        'connections': connections,
        'spanning_tree': spanning_tree,
        'tree_structure': tree_structure,
        'root_module': root_module,
        'analysis': analysis
    }


def extract_module_xml(original_xml_path, module_name, output_dir):
    """
    Extract a single module from the original XML file.
    
    Args:
        original_xml_path: Path to the original XML file
        module_name: Name of the module to extract
        output_dir: Directory to save the extracted module XML
    
    Returns:
        Path to the extracted module XML file
    """
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    
    # Find the module body
    module_body = None
    for body in root.iter('body'):
        if body.get('name') == module_name:
            module_body = body
            break
    
    if module_body is None:
        raise ValueError(f"Module {module_name} not found")
    
    # Create a new XML with just this module
    new_root = ET.Element('mujoco', model=f"{module_name}_module")
    
    # Copy relevant sections from original
    for section_name in ['compiler', 'option', 'asset']:
        section = root.find(section_name)
        if section is not None:
            new_root.append(copy.deepcopy(section))
    
    # Create worldbody with just this module
    worldbody = ET.SubElement(new_root, 'worldbody')
    worldbody.append(copy.deepcopy(module_body))
    
    # Copy actuators for this module
    actuator_section = root.find('actuator')
    if actuator_section is not None:
        new_actuator_section = ET.SubElement(new_root, 'actuator')
        for actuator in actuator_section:
            joint_name = actuator.get('joint')
            if joint_name and joint_name.startswith(module_name.replace('torso', 'joint')):
                new_actuator_section.append(copy.deepcopy(actuator))
    
    # Copy sensors for this module
    sensor_section = root.find('sensor')
    if sensor_section is not None:
        new_sensor_section = ET.SubElement(new_root, 'sensor')
        for sensor in sensor_section:
            sensor_name = sensor.get('name', '')
            if module_name in sensor_name:
                new_sensor_section.append(copy.deepcopy(sensor))
    
    # Save the extracted module
    output_path = Path(output_dir) / f"{module_name}_extracted.xml"
    ET.ElementTree(new_root).write(output_path, encoding='unicode', xml_declaration=True)
    
    print(f"Extracted module {module_name} to {output_path}")
    return str(output_path)


def update_keyframe_qpos(unified_root, original_root, root_module, modules, custom_qpos=None):
    """
    Update keyframe qpos to only include the root module's freejoint coordinates.
    
    Args:
        unified_root: The unified XML root element
        original_root: The original XML root element
        root_module: Name of the root module
        modules: Dictionary of all modules
        custom_qpos: Optional list/array of qpos values to use directly.
                    Should be length 7 + num_joints (7 for freejoint, rest for joints).
                    If None, extracts from original qpos and sets joints to 0.
    
    The original qpos has 7 values (3 pos + 4 quat) per freejoint for each module,
    followed by joint positions. We extract the root module's 7 values and set
    all joint positions to 0, unless custom_qpos is provided.
    """
    print("\n" + "=" * 60)
    print("UPDATING KEYFRAME QPOS")
    print("=" * 60)
    
    # Find keyframe section in original
    original_keyframe = original_root.find('keyframe')
    if original_keyframe:
    
        # Find keyframe section in unified (or create it)
        unified_keyframe = unified_root.find('keyframe')
        if unified_keyframe is None:
            unified_keyframe = ET.SubElement(unified_root, 'keyframe')
        else:
            # Clear existing keys
            for key in unified_keyframe.findall('key'):
                unified_keyframe.remove(key)
    
    # If custom_qpos is provided, use it directly
    if custom_qpos is not None:

        if not original_keyframe:
            # Create original_keyframe if it doesn't exist
            original_keyframe = ET.SubElement(original_root, 'keyframe')
            # Create a dummy key to extract structure
            dummy_key = ET.SubElement(original_keyframe, 'key')
            unified_keyframe = ET.SubElement(unified_root, 'keyframe')


        original_keys = original_keyframe.findall('key')


        print(f"  Using custom qpos with {len(custom_qpos)} values")
        
        # Get number of joints from original qpos
        original_key = original_keys[0]
        qpos_str = original_key.get('qpos', '')
        if qpos_str:
            qpos_values = [float(x) for x in qpos_str.split()]
            num_modules = len(modules)
            total_freejoint_values = num_modules * 7
            num_joints = len(qpos_values) - total_freejoint_values
            
            expected_length = 7 + num_joints
            if len(custom_qpos) != expected_length:
                print(f"  ⚠ Warning: Custom qpos length {len(custom_qpos)} doesn't match expected {expected_length}")
                print(f"    Expected: 7 freejoint + {num_joints} joints")
        
        # Process each key with the custom qpos
        for original_key in original_keys:
            key_name = original_key.get('name', 'unnamed')
            ctrl_str = original_key.get('ctrl', '')
            
            new_qpos_str = ' '.join([str(x) for x in custom_qpos])
            
            # Create new key element
            new_key = ET.SubElement(unified_keyframe, 'key')
            new_key.set('name', key_name)
            new_key.set('qpos', new_qpos_str)
            if ctrl_str:
                new_key.set('ctrl', ctrl_str)
            
            print(f"  ✓ Updated key '{key_name}' with custom qpos")
        
        print(f"  ✓ Processed {len(original_keys)} keyframe(s) with custom qpos")
        return
    

    if not original_keyframe:
        return
    # Default behavior: extract from original and set joints to 0
    # Determine the index of root_module in the module list
    # Modules appear in the order they are defined in the XML worldbody
    module_names = list(modules.keys())
    if root_module not in module_names:
        print(f"  ✗ Root module {root_module} not found in modules list")
        return
    
    root_module_index = module_names.index(root_module)
    print(f"  Root module '{root_module}' is at index {root_module_index} in module list: {module_names}")
    
    # Process each key
    for original_key in original_keys:
        key_name = original_key.get('name', 'unnamed')
        qpos_str = original_key.get('qpos', '')
        ctrl_str = original_key.get('ctrl', '')
        
        if not qpos_str:
            print(f"  Key '{key_name}': No qpos found, skipping")
            continue
        
        # Parse qpos values
        qpos_values = [float(x) for x in qpos_str.split()]
        num_modules = len(modules)
        
        # Each freejoint has 7 values (3 position + 4 quaternion)
        freejoint_values_per_module = 7
        total_freejoint_values = num_modules * freejoint_values_per_module
        
        if len(qpos_values) < total_freejoint_values:
            print(f"  ✗ Key '{key_name}': Expected at least {total_freejoint_values} values "
                  f"for {num_modules} freejoints, got {len(qpos_values)}")
            continue
        
        # Extract root module's freejoint values (7 values starting at root_module_index * 7)
        start_idx = root_module_index * freejoint_values_per_module
        end_idx = start_idx + freejoint_values_per_module
        root_freejoint_values = qpos_values[start_idx:end_idx]
        
        print(f"  Key '{key_name}':")
        print(f"    Original qpos length: {len(qpos_values)}")
        print(f"    Root module freejoint values (indices {start_idx}-{end_idx-1}): {root_freejoint_values}")
        
        # Joint values come after all freejoint values
        joint_values = qpos_values[total_freejoint_values:]
        num_joints = len(joint_values)
        
        # Create new qpos: root freejoint (7 values) + all joints set to 0
        new_qpos_values = root_freejoint_values + [0.0] * num_joints
        new_qpos_str = ' '.join([str(x) for x in new_qpos_values])
        
        print(f"    Number of joints: {num_joints}")
        print(f"    New qpos length: {len(new_qpos_values)} (7 freejoint + {num_joints} joints)")
        
        # Create new key element
        new_key = ET.SubElement(unified_keyframe, 'key')
        new_key.set('name', key_name)
        new_key.set('qpos', new_qpos_str)
        if ctrl_str:
            new_key.set('ctrl', ctrl_str)
        
        print(f"    ✓ Updated key '{key_name}' with new qpos")
    
    print(f"  ✓ Processed {len(original_keys)} keyframe(s)")


def restructure_modular_robot(xml_path, output_path=None, qpos=None):
    """
    Main function to restructure a modular robot by:
    1. Getting the reduction plan
    2. Transforming child modules
    3. Positioning modules using docking constraints
    4. Creating unified hierarchy
    5. Updating keyframe qpos to match new structure
    
    Args:
        xml_path: Path to the input XML file
        output_path: Optional path for the output XML file
        qpos: Optional custom qpos values for the keyframe. Should be a list/array
              with length 7 + num_joints (7 for freejoint: 3 pos + 4 quat, rest for joints).
              If None, extracts root module's freejoint from original and sets joints to 0.
              Example: [0, 0, 0.5, 1, 0, 0, 0, 0.1, -0.2, 0.3] for 3 joints
    """
    print("=" * 80)
    print("🤖 MODULAR ROBOT HIERARCHY RESTRUCTURING")
    print("=" * 80)
    
    if qpos is not None:
        print(f"📌 Custom qpos provided: {len(qpos)} values")
    
    # Step 1: Get reduction plan
    plan = get_reduction_plan(xml_path)
    
    root_module = plan['root_module']
    spanning_tree = plan['spanning_tree']
    modules = plan['modules']
    connections = plan['connections']
    original_tree = plan['tree']
    original_root = plan['root']
    
    print(f"\nReduction plan:")
    print(f"  Root module: {root_module}")
    print(f"  Child modules: {[child for parent, child, _, _, _ in spanning_tree]}")
    
    # Create output directory
    base_dir = Path(xml_path).parent
    work_dir = base_dir / "restructured_modules"
    work_dir.mkdir(exist_ok=True)
    
    # Step 2: Extract and transform each child module
    print("\n" + "=" * 60)
    print("STEP 2: TRANSFORMING CHILD MODULES")
    print("=" * 60)
    
    transformed_modules = {}
    
    for parent, child, site1, site2, weld in spanning_tree:
        print(f"\nProcessing connection: {parent} -> {child}")
        print(f"  Connection sites: {site1} <-> {site2}")
        
        # Extract the child module
        child_xml_path = extract_module_xml(xml_path, child, work_dir)
        
        # Determine which site belongs to the child module
        # Based on the naming pattern: dock-m{module_num}...
        child_module_num = child.replace('torso', '')
        child_site = site1 if f"m{child_module_num}" in site1 else site2
        
        print(f"  Child module {child} uses site: {child_site}")
        
        transformed_path = work_dir / f"{child}_transformed.xml"
        result = transform_module_hierarchy(
            child_xml_path, child, child_site, str(transformed_path)
        )
        
        if result:
            parent_site = site1 if child_site == site2 else site2
            transformed_modules[child] = {
                'path': result,
                'parent': parent,
                'parent_site': parent_site,
                'child_site': child_site,
                'weld': weld
            }
            print(f"  ✓ Child site: {child_site}, Parent site: {parent_site}")
    
    # Step 3: Calculate relative poses using get_B_in_A
    print("\n" + "=" * 60)
    print("STEP 3: CALCULATING RELATIVE POSES")
    print("=" * 60)
    
    module_poses = {}
    
    for child, info in transformed_modules.items():
        print(f"\nCalculating pose for {child} relative to {info['parent']}")
        
        # Get parent site pose
        parent_site = find_site_by_name(original_root, info['parent_site'])
        if parent_site is None:
            print(f"  ✗ Could not find parent site {info['parent_site']}")
            continue
        
        parent_pos, parent_quat = parse_site_pose(parent_site)
        print(f"  Parent site {info['parent_site']}: pos={parent_pos}, quat={parent_quat}")
        
        # Get child site pose (from transformed module)
        child_tree = ET.parse(info['path'])
        child_site = find_site_by_name(child_tree.getroot(), info['child_site'])
        if child_site is None:
            print(f"  ✗ Could not find child site {info['child_site']}")
            continue
        
        child_pos, child_quat = parse_site_pose(child_site)
        print(f"  Child site {info['child_site']}: pos={child_pos}, quat={child_quat}")
        
        # Calculate relative pose using get_B_in_A
        # We want child module (B) position in parent module (A) frame
        rel_pos, rel_quat = get_B_in_A(parent_pos, parent_quat, child_pos, child_quat)
        
        module_poses[child] = {
            'position': rel_pos,
            'quaternion': rel_quat,
            'info': info
        }
        
        print(f"  ✓ Calculated relative pose: pos={rel_pos}, quat={rel_quat}")
    
    # Step 4: Create unified hierarchy
    print("\n" + "=" * 60)
    print("STEP 4: CREATING UNIFIED HIERARCHY")
    print("=" * 60)
    
    # Start with the root module
    unified_tree = ET.parse(xml_path)
    unified_root = unified_tree.getroot()
    
    # Remove weld constraints (they're no longer needed)
    equality_section = unified_root.find('equality')
    if equality_section is not None:
        unified_root.remove(equality_section)
        print("  ✓ Removed weld constraints")
    
    # Remove child modules from their original locations
    worldbody = unified_root.find('worldbody')
    if worldbody is not None:
        for child in transformed_modules.keys():
            for body in worldbody.findall('body'):
                if body.get('name') == child:
                    worldbody.remove(body)
                    print(f"  ✓ Removed original {child} from worldbody")
    
        # Find the root module body
        root_body = None
        for body in worldbody.findall('body'):
            if body.get('name') == root_module:
                root_body = body
                break
    
    if root_body is None:
        raise ValueError(f"Could not find root module {root_module}")
    
    # Add transformed child modules as children of the body containing the parent docking site
    for child, pose_info in module_poses.items():
        parent_site_name = pose_info['info']['parent_site']
        parent_body = find_body_containing_site(unified_root, parent_site_name)
        
        if parent_body is None:
            print(f"  ✗ Could not find body containing parent site {parent_site_name}")
            continue
        
        parent_body_name = parent_body.get('name')
        print(f"\nAdding {child} as child of body {parent_body_name} (which contains site {parent_site_name})")
        
        # Load the transformed child module
        child_tree = ET.parse(pose_info['info']['path'])
        child_worldbody = child_tree.getroot().find('worldbody')
        child_body = None
        
        if child_worldbody is not None:
            child_body = child_worldbody.find('body')
        
        if child_body is not None:
            # Set the calculated pose
            pos = pose_info['position']
            quat = pose_info['quaternion']
            
            child_body.set('pos', ' '.join([str(x) for x in pos]))
            child_body.set('quat', ' '.join([str(x) for x in quat]))
            
            # Remove the freejoint (it's now a child, not root)
            freejoint = child_body.find('freejoint')
            if freejoint is not None:
                child_body.remove(freejoint)
                print(f"  ✓ Removed freejoint from {child}")
            
            # Add as child of the body containing the parent docking site
            parent_body.append(child_body)
            print(f"  ✓ Added {child} as child of {parent_body_name} at pos={pos}, quat={quat}")
    
    # Save the unified model
    if output_path is None:
        base_name = Path(xml_path).stem
        output_path = base_dir / f"{base_name}_unified.xml"
    
    # Step 5: Update keyframe qpos
    update_keyframe_qpos(unified_root, original_root, root_module, modules, custom_qpos=qpos)
    
    unified_tree.write(output_path, encoding='unicode', xml_declaration=True)
    
    print("\n" + "=" * 80)
    print("🎉 RESTRUCTURING COMPLETE!")
    print("=" * 80)
    print(f"✅ Unified model saved to: {output_path}")
    print(f"✅ Original modules: {len(modules)}")
    print(f"✅ Freejoints reduced from {len(modules)} to 1")
    print(f"✅ Weld constraints eliminated")
    print(f"✅ Keyframe qpos updated to match new structure")
    print(f"✅ New hierarchy structure:")
    print(f"   - Root module: {root_module}")
    for child, pose_info in module_poses.items():
        parent_site_name = pose_info['info']['parent_site']
        parent_body = find_body_containing_site(unified_root, parent_site_name)
        if parent_body is not None:
            parent_body_name = parent_body.get('name')
            print(f"   - {child} → attached to body '{parent_body_name}' (via site '{parent_site_name}')")
        else:
            print(f"   - {child} → attachment failed")
    
    return str(output_path)


def main():
    """Main function to run the restructuring process."""
    # xml_path = "/Users/chen/Lab/twist_controller/lab/chasing/restructure_bodies/tripod_cousin_simple.xml"
    # xml_path =  "/Users/chen/Lab/twist_controller/twist_controller/sim/assets/robots/tripod_cousin1.xml"
    # xml_path =  "/Users/chen/Lab/twist_controller/twist_controller/sim/assets/robots/test_lego2.xml"
    xml_path =  "/Users/chen/Lab/twist_controller/twist_controller/sim/assets/robots/test_lego2.xml"
    
    if not Path(xml_path).exists():
        print(f"Error: File {xml_path} does not exist")
        sys.exit(1)
    
    try:
        output_file = restructure_modular_robot(xml_path)

        
        print(f"\n🎊 SUCCESS! Restructured robot saved to:")
        print(f"   {output_file}")
        print(f"\nThe new model has a unified hierarchy without weld constraints!")
        
    except Exception as e:
        print(f"\n❌ Error during restructuring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
