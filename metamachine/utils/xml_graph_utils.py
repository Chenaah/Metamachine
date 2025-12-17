#!/usr/bin/env python3
"""
XML parsing and graph utilities for MuJoCo XML weld converter.

This module contains functions for parsing MuJoCo XML files, building
connection graphs, and creating spanning trees for module hierarchies.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict


def create_parent_map(root):
    """
    Create a mapping from child elements to their parent elements.
    
    Args:
        root: Root XML element
    
    Returns:
        dict: Mapping from child element to parent element
    """
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[child] = parent
    return parent_map


def find_module_bodies(root):
    """
    Find all bodies that represent modules (have freejoints).
    
    Args:
        root: Root XML element
    
    Returns:
        dict: module_name -> {body, freejoint, sites, child_bodies}
    """
    parent_map = create_parent_map(root)
    modules = {}
    
    freejoints = root.findall('.//freejoint')
    for freejoint in freejoints:
        # Find parent body of this freejoint
        current = freejoint
        while current is not None:
            current = parent_map.get(current)
            if current is not None and current.tag == 'body':
                module_name = current.get('name')
                modules[module_name] = {
                    'body': current,
                    'freejoint': freejoint,
                    'sites': [],
                    'child_bodies': []
                }
                break
    
    return modules


def map_sites_to_modules(root, modules):
    """
    Map all sites to their parent modules.
    
    Args:
        root: Root XML element
        modules: Dictionary of module information
    
    Returns:
        dict: site_name -> module_name
    """
    parent_map = create_parent_map(root)
    sites = root.findall('.//site')
    site_to_module = {}
    
    for site in sites:
        site_name = site.get('name')
        
        # Find which module this site belongs to by traversing up the hierarchy
        current = site
        site_module = None
        site_body = None
        
        while current is not None:
            current = parent_map.get(current)
            if current is not None and current.tag == 'body':
                site_body = current.get('name')
                # Check if this body is a module (has freejoint)
                if site_body in modules:
                    site_module = site_body
                    break
                # Otherwise, check if this body belongs to a module
                body_current = current
                while body_current is not None:
                    body_current = parent_map.get(body_current)
                    if body_current is not None and body_current.tag == 'body':
                        body_name = body_current.get('name')
                        if body_name in modules:
                            site_module = body_name
                            break
                if site_module:
                    break
        
        if site_module:
            site_to_module[site_name] = site_module
            modules[site_module]['sites'].append({
                'name': site_name,
                'element': site,
                'body': site_body,
                'pos': site.get('pos', '0 0 0')
            })
    
    return site_to_module


def build_connection_graph(root, modules, site_to_module):
    """
    Build a connection graph between modules based on weld constraints.
    
    Args:
        root: Root XML element
        modules: Dictionary of module information
        site_to_module: Mapping from site names to module names
    
    Returns:
        dict: module -> [(connected_module, site1, site2, weld_element), ...]
    """
    connections = defaultdict(list)
    
    equality_section = root.find('.//equality')
    if equality_section is not None:
        welds = equality_section.findall('weld')
        
        for weld in welds:
            site1_name = weld.get('site1')
            site2_name = weld.get('site2')
            
            module1 = site_to_module.get(site1_name)
            module2 = site_to_module.get(site2_name)
            
            if module1 and module2 and module1 != module2:
                # Add bidirectional connections
                connections[module1].append((module2, site1_name, site2_name, weld))
                connections[module2].append((module1, site2_name, site1_name, weld))
    
    return dict(connections)


def select_root_module(connections):
    """
    Select the best root module for the spanning tree (most connected).
    
    Args:
        connections: Connection graph dictionary
    
    Returns:
        str: Name of the selected root module
    """
    if not connections:
        return None
    
    # Count connections for each module
    connection_counts = {module: len(conn_list) for module, conn_list in connections.items()}
    
    # Select most connected module as root
    root_module = max(connection_counts.keys(), key=lambda x: connection_counts[x])
    return root_module


def build_spanning_tree(connections, root_module):
    """
    Build a spanning tree from the connection graph using DFS.
    
    Args:
        connections: Connection graph dictionary
        root_module: Name of the root module
    
    Returns:
        tuple: (spanning_tree, tree_structure)
            spanning_tree: List of (parent, child, site1, site2, weld) edges
            tree_structure: Dict of module -> [child_modules]
    """
    if not root_module:
        return [], {}
    
    visited = set()
    spanning_tree = []
    tree_structure = {}
    
    def dfs(current_module):
        if current_module in visited:
            return
        
        visited.add(current_module)
        tree_structure[current_module] = []
        
        # Explore all connections from current module
        if current_module in connections:
            for neighbor_module, site1, site2, weld in connections[current_module]:
                if neighbor_module not in visited:
                    spanning_tree.append((current_module, neighbor_module, site1, site2, weld))
                    tree_structure[current_module].append(neighbor_module)
                    dfs(neighbor_module)
    
    dfs(root_module)
    return spanning_tree, tree_structure


def print_tree_structure(tree_structure, root_module, indent=0):
    """
    Print a hierarchical tree structure.
    
    Args:
        tree_structure: Dict of module -> [child_modules]
        root_module: Name of the root module
        indent: Current indentation level
    """
    print("  " * indent + f"- {root_module}")
    for child in tree_structure.get(root_module, []):
        print_tree_structure(tree_structure, child, indent + 1)


def analyze_freejoint_reduction(modules, spanning_tree):
    """
    Analyze how many freejoints will be removed after conversion.
    
    Args:
        modules: Dictionary of module information
        spanning_tree: List of spanning tree edges
    
    Returns:
        dict: Analysis results with counts and lists
    """
    all_modules = set(modules.keys())
    connected_modules = set()
    
    # Find all modules in the spanning tree
    for parent, child, _, _, _ in spanning_tree:
        connected_modules.add(parent)
        connected_modules.add(child)
    
    disconnected_modules = all_modules - connected_modules
    
    original_freejoints = len(modules)
    remaining_freejoints = len(disconnected_modules) + 1  # root keeps its freejoint
    removed_freejoints = original_freejoints - remaining_freejoints
    
    return {
        'original_count': original_freejoints,
        'remaining_count': remaining_freejoints,
        'removed_count': removed_freejoints,
        'connected_modules': connected_modules,
        'disconnected_modules': disconnected_modules
    }
