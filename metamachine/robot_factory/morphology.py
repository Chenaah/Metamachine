"""
Unified Robot Morphology Representation

This module provides a graph-based morphology representation that works
across different robot factories. It supports both the tree-based modular_legs
approach and the more flexible graph-based lego_legs approach.

The key abstraction is RobotGraph, which represents:
- Nodes: Components (balls, sticks, adaptors, etc.)
- Edges: Connections between component docking sites

This design allows:
- Easy conversion between different factory formats
- Graph-based robot generation
- Serialization for storage/transmission
- Mutation operations for evolutionary algorithms

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Optional, TypeVar, Generic


class ComponentType(Enum):
    """Types of components that can be used in a robot."""
    BALL = "ball"
    PASSIVE_BALL = "passive_ball"
    STICK = "stick"
    ADAPTOR = "adaptor"
    ADAPTOR4 = "adaptor4"
    STICK4 = "stick4"
    CUSTOM = "custom"


@dataclass
class ComponentSpec:
    """
    Specification for a robot component.
    
    Attributes:
        component_type: Type of the component
        name: Unique name for the component (auto-generated if not provided)
        params: Type-specific parameters (length, mass, color, etc.)
        docking_sites: List of docking site names this component exposes
    """
    component_type: ComponentType
    name: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    docking_sites: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default docking sites based on component type."""
        if not self.docking_sites:
            self.docking_sites = self._get_default_docking_sites()
    
    def _get_default_docking_sites(self) -> list[str]:
        """Get default docking sites for this component type."""
        if self.component_type in (ComponentType.BALL, ComponentType.PASSIVE_BALL):
            return ["0f", "1f"]  # Two hemispheres
        elif self.component_type in (ComponentType.STICK, ComponentType.STICK4):
            return ["0m", "1m"]  # Two ends (male connectors)
        elif self.component_type == ComponentType.ADAPTOR4:
            return ["0f", "0'f", "1f", "1'f", "2f", "2'f"]  # 6 connection points
        elif self.component_type == ComponentType.ADAPTOR:
            return ["0f", "1f", "2f", "3f"]  # 4 connection points
        return []
    
    def get_site_name(self, site_id: str) -> str:
        """Get the full site name for a docking site."""
        if self.name:
            return f"dock-{self.name}-{site_id}"
        return site_id
    
    @classmethod
    def ball(
        cls,
        name: Optional[str] = None,
        add_sites: bool = True,
        passive: bool = False,
        color: Optional[tuple[float, float, float]] = None,
        **kwargs,
    ) -> "ComponentSpec":
        """Factory method to create a ball component."""
        comp_type = ComponentType.PASSIVE_BALL if passive else ComponentType.BALL
        params = {"add_sites": add_sites, "passive": passive}
        if color is not None:
            params["color"] = color
        params.update(kwargs)
        return cls(component_type=comp_type, name=name, params=params)
    
    @classmethod
    def stick(
        cls,
        name: Optional[str] = None,
        length: float = 0.075,
        mass: float = 0.128,
        dock_offset_angle: float = 0.0,
        **kwargs,
    ) -> "ComponentSpec":
        """Factory method to create a stick component."""
        params = {
            "length": length,
            "mass": mass,
            "dock_offset_angle": dock_offset_angle,
        }
        params.update(kwargs)
        return cls(component_type=ComponentType.STICK4, name=name, params=params)
    
    @classmethod
    def adaptor(
        cls,
        name: Optional[str] = None,
        mass: float = 0.26,
        **kwargs,
    ) -> "ComponentSpec":
        """Factory method to create an adaptor component."""
        params = {"mass": mass}
        params.update(kwargs)
        return cls(component_type=ComponentType.ADAPTOR4, name=name, params=params)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "component_type": self.component_type.value,
            "name": self.name,
            "params": self.params,
            "docking_sites": self.docking_sites,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComponentSpec":
        """Deserialize from dictionary."""
        return cls(
            component_type=ComponentType(data["component_type"]),
            name=data.get("name"),
            params=data.get("params", {}),
            docking_sites=data.get("docking_sites", []),
        )


@dataclass
class Connection:
    """
    Represents a weld connection between two component docking sites.
    
    Attributes:
        source_component: Name of the source component
        source_site: Docking site on the source component
        target_component: Name of the target component
        target_site: Docking site on the target component
        params: Additional connection parameters (solref, solimp, etc.)
    """
    source_component: str
    source_site: str
    target_component: str
    target_site: str
    params: dict[str, Any] = field(default_factory=dict)
    
    def get_source_site_name(self) -> str:
        """Get the full MuJoCo site name for the source."""
        return f"dock-{self.source_component}-{self.source_site}"
    
    def get_target_site_name(self) -> str:
        """Get the full MuJoCo site name for the target."""
        return f"dock-{self.target_component}-{self.target_site}"
    
    def to_site_pair(self) -> tuple[str, str]:
        """Convert to a site pair tuple for weld constraints."""
        return (self.get_source_site_name(), self.get_target_site_name())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_component": self.source_component,
            "source_site": self.source_site,
            "target_component": self.target_component,
            "target_site": self.target_site,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Connection":
        """Deserialize from dictionary."""
        return cls(
            source_component=data["source_component"],
            source_site=data["source_site"],
            target_component=data["target_component"],
            target_site=data["target_site"],
            params=data.get("params", {}),
        )


class RobotGraph:
    """
    Graph-based representation of a robot morphology.
    
    This class represents robots as a graph where:
    - Nodes are components (balls, sticks, adaptors)
    - Edges are weld connections between docking sites
    
    This unified representation can be used by different robot factories
    and supports operations like:
    - Adding/removing components
    - Adding/removing connections
    - Serialization/deserialization
    - Conversion to factory-specific formats
    
    Example:
        >>> graph = RobotGraph()
        >>> graph.add_component(ComponentSpec.ball(name="m0", add_sites=True))
        >>> graph.add_component(ComponentSpec.ball(name="m1", add_sites=True))
        >>> graph.add_component(ComponentSpec.adaptor(name="belly"))
        >>> graph.add_component(ComponentSpec.stick(name="thigh0", length=0.075))
        >>> graph.connect("belly", "0f", "thigh0", "1m")
        >>> graph.connect("m0", "0f", "thigh0", "0m")
    """
    
    def __init__(
        self,
        components: Optional[list[ComponentSpec]] = None,
        connections: Optional[list[Connection]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize a RobotGraph.
        
        Args:
            components: Initial list of components
            connections: Initial list of connections
            metadata: Optional metadata (robot name, description, etc.)
        """
        self._components: dict[str, ComponentSpec] = {}
        self._connections: list[Connection] = []
        self._component_counter = 0
        self.metadata = metadata or {}
        
        # Add initial components and connections
        if components:
            for comp in components:
                self.add_component(comp)
        if connections:
            for conn in connections:
                self._connections.append(conn)
    
    @property
    def components(self) -> dict[str, ComponentSpec]:
        """Get all components by name."""
        return self._components.copy()
    
    @property
    def connections(self) -> list[Connection]:
        """Get all connections."""
        return self._connections.copy()
    
    @property
    def num_components(self) -> int:
        """Get the number of components."""
        return len(self._components)
    
    @property
    def num_connections(self) -> int:
        """Get the number of connections."""
        return len(self._connections)
    
    def add_component(self, spec: ComponentSpec) -> str:
        """
        Add a component to the graph.
        
        Args:
            spec: Component specification
            
        Returns:
            The name of the added component
        """
        # Auto-generate name if not provided
        if spec.name is None:
            spec.name = f"c{self._component_counter}"
            self._component_counter += 1
        
        if spec.name in self._components:
            raise ValueError(f"Component '{spec.name}' already exists")
        
        self._components[spec.name] = spec
        return spec.name
    
    def remove_component(self, name: str) -> bool:
        """
        Remove a component and its connections.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._components:
            return False
        
        # Remove all connections involving this component
        self._connections = [
            conn for conn in self._connections
            if conn.source_component != name and conn.target_component != name
        ]
        
        del self._components[name]
        return True
    
    def get_component(self, name: str) -> Optional[ComponentSpec]:
        """Get a component by name."""
        return self._components.get(name)
    
    def connect(
        self,
        source_component: str,
        source_site: str,
        target_component: str,
        target_site: str,
        **params,
    ) -> Connection:
        """
        Add a connection between two components.
        
        Args:
            source_component: Name of the source component
            source_site: Docking site on the source
            target_component: Name of the target component
            target_site: Docking site on the target
            **params: Additional connection parameters
            
        Returns:
            The created Connection object
        """
        # Validate components exist
        if source_component not in self._components:
            raise ValueError(f"Source component '{source_component}' not found")
        if target_component not in self._components:
            raise ValueError(f"Target component '{target_component}' not found")
        
        conn = Connection(
            source_component=source_component,
            source_site=source_site,
            target_component=target_component,
            target_site=target_site,
            params=params,
        )
        self._connections.append(conn)
        return conn
    
    def disconnect(
        self,
        source_component: str,
        source_site: str,
        target_component: str,
        target_site: str,
    ) -> bool:
        """
        Remove a connection.
        
        Returns:
            True if removed, False if not found
        """
        for i, conn in enumerate(self._connections):
            if (conn.source_component == source_component and
                conn.source_site == source_site and
                conn.target_component == target_component and
                conn.target_site == target_site):
                self._connections.pop(i)
                return True
        return False
    
    def get_connections_for(self, component_name: str) -> list[Connection]:
        """Get all connections involving a component."""
        return [
            conn for conn in self._connections
            if conn.source_component == component_name or conn.target_component == component_name
        ]
    
    def get_weld_pairs(self) -> list[tuple[str, str]]:
        """Get all connections as site pairs for weld constraints."""
        return [conn.to_site_pair() for conn in self._connections]
    
    def get_components_by_type(self, comp_type: ComponentType) -> list[ComponentSpec]:
        """Get all components of a specific type."""
        return [
            comp for comp in self._components.values()
            if comp.component_type == comp_type
        ]
    
    def iter_components(self) -> Iterator[tuple[str, ComponentSpec]]:
        """Iterate over components as (name, spec) pairs."""
        yield from self._components.items()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a dictionary."""
        return {
            "components": [comp.to_dict() for comp in self._components.values()],
            "connections": [conn.to_dict() for conn in self._connections],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotGraph":
        """Deserialize a graph from a dictionary."""
        components = [ComponentSpec.from_dict(c) for c in data.get("components", [])]
        connections = [Connection.from_dict(c) for c in data.get("connections", [])]
        return cls(
            components=components,
            connections=connections,
            metadata=data.get("metadata", {}),
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize the graph to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RobotGraph":
        """Deserialize a graph from JSON."""
        return cls.from_dict(json.loads(json_str))
    
    def copy(self) -> "RobotGraph":
        """Create a deep copy of this graph."""
        return RobotGraph.from_dict(self.to_dict())
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the graph structure.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check for empty graph
        if not self._components:
            errors.append("Graph has no components")
        
        # Check connections reference existing components and sites
        for conn in self._connections:
            if conn.source_component not in self._components:
                errors.append(f"Connection references unknown component: {conn.source_component}")
            if conn.target_component not in self._components:
                errors.append(f"Connection references unknown component: {conn.target_component}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def from_legacy_sequence(cls, sequence: list[int]) -> "RobotGraph":
        """
        Create a RobotGraph from a legacy modular_legs sequence.
        
        The sequence format is: [parent_id, parent_dock, child_dock, orientation, ...]
        Each group of 4 integers defines one connection.
        
        Args:
            sequence: Legacy sequence format
            
        Returns:
            RobotGraph representation
        """
        graph = RobotGraph()
        
        if not sequence:
            return graph
        
        if len(sequence) % 4 != 0:
            raise ValueError(f"Sequence length must be multiple of 4, got {len(sequence)}")
        
        # First, create all the modules needed
        # Module 0 is always created first, then each connection adds a new module
        num_modules = len(sequence) // 4 + 1
        for i in range(num_modules):
            graph.add_component(ComponentSpec.ball(name=f"m{i}", add_sites=True))
        
        # Then create connections based on the sequence
        # Note: This is a simplified conversion - actual dock mapping may vary
        for i in range(0, len(sequence), 4):
            parent_id = sequence[i]
            parent_dock = sequence[i + 1]
            child_dock = sequence[i + 2]
            # orientation = sequence[i + 3]  # Not used in simple conversion
            
            child_id = i // 4 + 1
            
            # Map dock IDs to site names (simplified mapping)
            parent_site = f"{parent_dock % 2}f"
            child_site = f"{child_dock % 2}f"
            
            graph.connect(f"m{parent_id}", parent_site, f"m{child_id}", child_site)
        
        return graph


# Convenience functions for creating common robot configurations

def create_tripod_graph(
    thigh_length: float = 0.075,
    calf_length: float = 0.15,
) -> RobotGraph:
    """
    Create a standard tripod robot graph.
    
    Args:
        thigh_length: Length of thigh sticks
        calf_length: Length of calf sticks
        
    Returns:
        RobotGraph for a tripod
    """
    graph = RobotGraph(metadata={"name": "tripod", "type": "lego_legs"})
    
    # Add ball modules (hip joints)
    for i in range(3):
        graph.add_component(ComponentSpec.ball(name=f"m{i}", add_sites=True))
    
    # Add central adaptor (belly)
    graph.add_component(ComponentSpec.adaptor(name="belly", mass=0.26))
    
    # Add thigh sticks
    for i in range(3):
        graph.add_component(ComponentSpec.stick(
            name=f"thigh{i}",
            length=thigh_length,
            mass=0.128,
        ))
    
    # Add calf sticks
    for i in range(3):
        graph.add_component(ComponentSpec.stick(
            name=f"calf{i}",
            length=calf_length,
            mass=0.126,
        ))
    
    # Add connections for each leg
    for i in range(3):
        # Belly to thigh
        graph.connect("belly", f"{i}f", f"thigh{i}", "1m")
        # Ball to thigh
        graph.connect(f"m{i}", "0f", f"thigh{i}", "0m")
        # Calf to ball
        graph.connect(f"calf{i}", "0m", f"m{i}", "1f")
    
    return graph


def create_quadruped_graph(
    thigh_length: float = 0.075,
    calf_length: float = 0.15,
) -> RobotGraph:
    """
    Create a standard quadruped robot graph.
    
    Args:
        thigh_length: Length of thigh sticks
        calf_length: Length of calf sticks
        
    Returns:
        RobotGraph for a quadruped
    """
    graph = RobotGraph(metadata={"name": "quadruped", "type": "lego_legs"})
    
    # Add ball modules (hip joints) - 4 for quadruped
    for i in range(4):
        graph.add_component(ComponentSpec.ball(name=f"m{i}", add_sites=True))
    
    # Add central adaptor (belly)
    graph.add_component(ComponentSpec.adaptor(name="belly", mass=0.30))
    
    # Add thigh sticks
    for i in range(4):
        graph.add_component(ComponentSpec.stick(
            name=f"thigh{i}",
            length=thigh_length,
            mass=0.128,
        ))
    
    # Add calf sticks
    for i in range(4):
        graph.add_component(ComponentSpec.stick(
            name=f"calf{i}",
            length=calf_length,
            mass=0.126,
        ))
    
    # Add connections for each leg
    belly_sites = ["0f", "1f", "1'f", "2f"]  # 4 different sites on adaptor
    for i in range(4):
        # Belly to thigh
        graph.connect("belly", belly_sites[i], f"thigh{i}", "1m")
        # Ball to thigh
        graph.connect(f"m{i}", "0f", f"thigh{i}", "0m")
        # Calf to ball
        graph.connect(f"calf{i}", "0m", f"m{i}", "1f")
    
    return graph

