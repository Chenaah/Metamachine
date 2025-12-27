"""
Abstract Random Robot Generation API

This module provides the ABSTRACT layer for random robot generation that can
be used by any robot factory. It defines:
- Generic dock system (DockGender, DockSpec, ComponentDockRegistry)
- Abstract budget system (ComponentBudget)
- Generation capability mixin (RandomGenerationCapability)
- Utility functions for graph visualization and serialization

Factory-specific implementations (like lego_legs) should:
1. Define their own dock registry
2. Implement their own RandomGraphGenerator subclass
3. Provide factory-specific budget helpers

Example (for a plugin factory):
    from metamachine.robot_factory.random_generation import (
        ComponentDockRegistry,
        DockSpec,
        DockGender,
        ComponentBudget,
        RandomGenerationCapability,
    )
    
    # Create factory-specific registry
    MY_DOCK_REGISTRY = ComponentDockRegistry()
    MY_DOCK_REGISTRY.register_component(MyComponentType.JOINT, [...])
    
    # Implement factory with random generation
    class MyFactory(BaseRobotFactory, RandomGenerationCapability):
        def get_dock_registry(self):
            return MY_DOCK_REGISTRY

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from .morphology import ComponentSpec, ComponentType, Connection, RobotGraph


# =============================================================================
# Dock System (Generic)
# =============================================================================

class DockGender(Enum):
    """Gender of a dock connector."""
    MALE = "m"
    FEMALE = "f"
    NEUTRAL = "n"  # Can connect to either


@dataclass
class DockSpec:
    """
    Specification for a docking site.
    
    Attributes:
        name: Dock identifier (e.g., "0f", "1m")
        gender: Gender of the connector (male/female/neutral)
        position: Optional position index for ordering
        compatible_with: List of dock names this can connect to (if restricted)
    """
    name: str
    gender: DockGender
    position: int = 0
    compatible_with: Optional[list[str]] = None
    
    @classmethod
    def male(cls, name: str, position: int = 0) -> "DockSpec":
        """Create a male dock."""
        return cls(name=name, gender=DockGender.MALE, position=position)
    
    @classmethod
    def female(cls, name: str, position: int = 0) -> "DockSpec":
        """Create a female dock."""
        return cls(name=name, gender=DockGender.FEMALE, position=position)
    
    @classmethod
    def from_string(cls, dock_str: str) -> "DockSpec":
        """
        Parse a dock specification from string format.
        
        Supported formats:
        - "0f" -> female dock at position 0
        - "1m" -> male dock at position 1
        - "0'f" -> female dock at position 0' (alternate)
        """
        if dock_str.endswith("f"):
            gender = DockGender.FEMALE
            name_part = dock_str[:-1]
        elif dock_str.endswith("m"):
            gender = DockGender.MALE
            name_part = dock_str[:-1]
        else:
            gender = DockGender.NEUTRAL
            name_part = dock_str
        
        # Extract position number
        try:
            position = int(name_part.replace("'", ""))
        except ValueError:
            position = 0
        
        return cls(name=dock_str, gender=gender, position=position)


@dataclass
class ComponentDockRegistry:
    """
    Registry of available docks for each component type.
    
    This defines what docking sites are available on each component type
    and their connection rules. Each factory should create its own registry.
    """
    docks: dict[ComponentType, list[DockSpec]] = field(default_factory=dict)
    connection_rules: Optional[Callable[[DockSpec, DockSpec], bool]] = None
    
    def register_component(
        self,
        component_type: ComponentType,
        docks: list[DockSpec],
    ) -> None:
        """Register docks for a component type."""
        self.docks[component_type] = docks
    
    def get_docks(self, component_type: ComponentType) -> list[DockSpec]:
        """Get available docks for a component type."""
        return self.docks.get(component_type, [])
    
    def get_male_docks(self, component_type: ComponentType) -> list[DockSpec]:
        """Get male docks for a component type."""
        return [d for d in self.get_docks(component_type) if d.gender == DockGender.MALE]
    
    def get_female_docks(self, component_type: ComponentType) -> list[DockSpec]:
        """Get female docks for a component type."""
        return [d for d in self.get_docks(component_type) if d.gender == DockGender.FEMALE]
    
    def can_connect(self, dock1: DockSpec, dock2: DockSpec) -> bool:
        """Check if two docks can connect."""
        if self.connection_rules:
            return self.connection_rules(dock1, dock2)
        
        # Default rule: male connects to female
        if dock1.gender == DockGender.MALE and dock2.gender == DockGender.FEMALE:
            return True
        if dock1.gender == DockGender.FEMALE and dock2.gender == DockGender.MALE:
            return True
        if dock1.gender == DockGender.NEUTRAL or dock2.gender == DockGender.NEUTRAL:
            return True
        return False


# =============================================================================
# Component Budget (Generic)
# =============================================================================

@dataclass
class ComponentBudget:
    """
    Budget constraints for component types.
    
    Defines how many of each component type should be used
    when generating random robots. This is a generic base class;
    factory-specific helpers can be added in plugins.
    """
    components: dict[ComponentType, int] = field(default_factory=dict)
    min_legs: int = 2
    
    def __post_init__(self):
        # Ensure all values are ints
        self.components = {k: int(v) for k, v in self.components.items()}
    
    @classmethod
    def create(
        cls,
        components: dict[ComponentType, int],
        min_legs: int = 2,
    ) -> "ComponentBudget":
        """Create a budget with explicit component counts."""
        return cls(components=components, min_legs=min_legs)
    
    def total_components(self) -> int:
        """Get total number of components."""
        return sum(self.components.values())
    
    def get_count(self, comp_type: ComponentType) -> int:
        """Get count for a component type."""
        return self.components.get(comp_type, 0)
    
    def set_count(self, comp_type: ComponentType, count: int) -> None:
        """Set count for a component type."""
        self.components[comp_type] = count


# =============================================================================
# Generation Configuration (Generic)
# =============================================================================

class GenerationMode(Enum):
    """Mode for random generation."""
    STRUCTURED = "structured"  # Creates proper structured morphologies
    RANDOM = "random"          # Fully random connections


@dataclass
class GenerationConfig:
    """
    Configuration for random robot generation.
    
    This is a generic base config. Factory-specific configs can
    extend this with additional parameters.
    """
    mode: GenerationMode = GenerationMode.STRUCTURED
    connection_density: float = 0.75  # Target % of possible connections
    seed: Optional[int] = None
    
    # Generic length ranges (can be overridden)
    length_range: tuple[float, float] = (0.15, 0.30)
    
    # Additional factory-specific params stored here
    extra_params: dict = field(default_factory=dict)


# =============================================================================
# Abstract Random Graph Generator
# =============================================================================

class BaseRandomGraphGenerator(ABC):
    """
    Abstract base class for random graph generators.
    
    Each factory should implement its own subclass with
    factory-specific generation logic.
    """
    
    def __init__(
        self,
        budget: ComponentBudget,
        dock_registry: ComponentDockRegistry,
        config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize the generator.
        
        Args:
            budget: Component budget constraints
            dock_registry: Registry defining docks for each component type
            config: Generation configuration
        """
        self.budget = budget
        self.dock_registry = dock_registry
        self.config = config or GenerationConfig()
        
        if self.config.seed is not None:
            random.seed(self.config.seed)
    
    @abstractmethod
    def generate(self) -> RobotGraph:
        """
        Generate a random robot graph.
        
        Returns:
            A valid RobotGraph with connected components
        """
        pass
    
    def generate_batch(self, count: int) -> list[RobotGraph]:
        """Generate multiple random robot graphs."""
        return [self.generate() for _ in range(count)]


# =============================================================================
# Factory Integration (Abstract Mixin)
# =============================================================================

class RandomGenerationCapability(ABC):
    """
    Abstract mixin for factories that support random generation.
    
    Add this to a factory class to enable random robot generation:
    
        class MyFactory(BaseRobotFactory, RandomGenerationCapability):
            def get_dock_registry(self):
                return MY_DOCK_REGISTRY
            
            def create_random_generator(self, budget, config):
                return MyRandomGraphGenerator(budget, self.get_dock_registry(), config)
    """
    
    @abstractmethod
    def get_dock_registry(self) -> ComponentDockRegistry:
        """Get the dock registry for this factory's component types."""
        pass
    
    @abstractmethod
    def create_random_generator(
        self,
        budget: Optional[ComponentBudget] = None,
        config: Optional[GenerationConfig] = None,
    ) -> BaseRandomGraphGenerator:
        """
        Create a random generator configured for this factory.
        
        Args:
            budget: Component budget (uses factory default if None)
            config: Generation configuration
            
        Returns:
            Configured random graph generator
        """
        pass
    
    def generate_random_morphology(
        self,
        budget: Optional[ComponentBudget] = None,
        config: Optional[GenerationConfig] = None,
    ) -> RobotGraph:
        """
        Generate a random morphology graph.
        
        Args:
            budget: Component budget
            config: Generation configuration
            
        Returns:
            A RobotGraph representing the random morphology
        """
        generator = self.create_random_generator(budget, config)
        return generator.generate()
    
    def generate_random_robot(
        self,
        budget: Optional[ComponentBudget] = None,
        config: Optional[GenerationConfig] = None,
        **create_kwargs,
    ):
        """
        Generate and create a random robot in one step.
        
        Args:
            budget: Component budget
            config: Generation configuration
            **create_kwargs: Additional arguments for create_robot
            
        Returns:
            A robot instance created from the random graph
        """
        graph = self.generate_random_morphology(budget, config)
        return self.create_robot(morphology=graph, **create_kwargs)


# =============================================================================
# Utility Functions (Generic)
# =============================================================================

def visualize_graph(graph: RobotGraph, output_path: Optional[str] = None) -> str:
    """
    Generate a DOT representation of the robot graph.
    
    Args:
        graph: The robot graph to visualize
        output_path: Optional path to save the DOT file
        
    Returns:
        DOT format string
    """
    lines = ["digraph robot {"]
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box];")
    
    # Add nodes with labels
    for name, comp in graph.components.items():
        label = f"{name}\\n({comp.component_type.value})"
        color = {
            ComponentType.BALL: "lightblue",
            ComponentType.PASSIVE_BALL: "lightgray",
            ComponentType.STICK4: "lightgreen",
            ComponentType.STICK: "lightgreen",
            ComponentType.ADAPTOR4: "lightyellow",
            ComponentType.ADAPTOR: "lightyellow",
        }.get(comp.component_type, "white")
        lines.append(f'  {name} [label="{label}", fillcolor="{color}", style=filled];')
    
    # Add edges
    for conn in graph.connections:
        label = f"{conn.source_site} -> {conn.target_site}"
        lines.append(f'  {conn.source_component} -> {conn.target_component} [label="{label}"];')
    
    lines.append("}")
    
    dot_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(dot_str)
    
    return dot_str


def graph_to_yaml_dict(graph: RobotGraph) -> dict:
    """
    Convert a RobotGraph to a YAML-compatible dictionary.
    
    Args:
        graph: The robot graph
        
    Returns:
        Dictionary suitable for YAML serialization
    """
    components = []
    for name, comp in graph.components.items():
        comp_dict = {
            "component_type": comp.component_type.value,
            "name": name,
            "params": comp.params,
        }
        components.append(comp_dict)
    
    connections = []
    for conn in graph.connections:
        conn_dict = {
            "source_component": conn.source_component,
            "source_site": conn.source_site,
            "target_component": conn.target_component,
            "target_site": conn.target_site,
            "params": conn.params,
        }
        connections.append(conn_dict)
    
    return {
        "metadata": graph.metadata,
        "components": components,
        "connections": connections,
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Dock system
    "DockGender",
    "DockSpec",
    "ComponentDockRegistry",
    # Budget
    "ComponentBudget",
    # Generation
    "GenerationMode",
    "GenerationConfig",
    "BaseRandomGraphGenerator",
    # Factory integration
    "RandomGenerationCapability",
    # Utilities
    "visualize_graph",
    "graph_to_yaml_dict",
]
