"""
Configuration management for inverse kinematics solver.

This module provides configuration classes and utilities for managing
solver parameters, robot models, and evaluation settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import yaml
import os
import numpy as np
from pathlib import Path


@dataclass
class RobotConfig:
    """Configuration for robot models."""
    
    name: str
    type: str  # 'planar_2r', 'planar_3r', 'spatial_6r'
    link_lengths: Optional[Tuple[float, ...]] = None
    joint_limits: Optional[List[Tuple[float, float]]] = None
    dh_params: Optional[List[Tuple[float, float, float, float]]] = None
    base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SolverConfig:
    """Configuration for IK solvers."""
    
    name: str
    type: str  # 'analytical_2r', 'newton_raphson', etc.
    tolerance: float = 1e-6
    max_iterations: int = 100
    damping: float = 0.01
    step_size: float = 0.1
    initial_lambda: float = 0.01
    lambda_multiplier: float = 10.0
    null_space_gain: float = 0.1


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and benchmarking."""
    
    n_test_points: int = 1000
    test_distributions: List[str] = field(default_factory=lambda: ['uniform', 'gaussian', 'edge_case'])
    workspace_bounds: Optional[Tuple[List[float], List[float]]] = None
    performance_metrics: List[str] = field(default_factory=lambda: [
        'success_rate', 'average_error', 'average_time', 'workspace_coverage', 'robustness_score'
    ])


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    
    gui: bool = True
    timestep: float = 1/240
    physics_engine: str = 'pybullet'
    ground_plane: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    backend: str = 'streamlit'  # 'streamlit', 'gradio', 'matplotlib'
    theme: str = 'light'
    figure_size: Tuple[int, int] = (800, 600)
    dpi: int = 100
    save_plots: bool = True
    plot_format: str = 'png'


@dataclass
class IKConfig:
    """Main configuration class for the IK solver system."""
    
    robot: RobotConfig
    solver: SolverConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    random_seed: int = 42
    log_level: str = 'INFO'
    output_dir: str = 'output'
    cache_dir: str = 'cache'


class ConfigManager:
    """Manager for configuration files and settings."""
    
    def __init__(self, config_dir: str = 'config'):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self, config_file: str) -> IKConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Loaded configuration object
        """
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._dict_to_config(config_data)
    
    def save_config(self, config: IKConfig, config_file: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            config_file: Path to configuration file
        """
        config_path = self.config_dir / config_file
        
        config_data = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def create_default_config(self, config_file: str = 'default.yaml') -> IKConfig:
        """
        Create and save default configuration.
        
        Args:
            config_file: Name of configuration file
            
        Returns:
            Default configuration object
        """
        # Default robot configuration
        robot_config = RobotConfig(
            name='planar_2r_robot',
            type='planar_2r',
            link_lengths=(1.0, 1.0),
            joint_limits=[(-np.pi, np.pi), (-np.pi, np.pi)]
        )
        
        # Default solver configuration
        solver_config = SolverConfig(
            name='analytical_2r_solver',
            type='analytical_2r',
            tolerance=1e-6,
            max_iterations=100
        )
        
        # Create main configuration
        config = IKConfig(
            robot=robot_config,
            solver=solver_config
        )
        
        # Save to file
        self.save_config(config, config_file)
        
        return config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> IKConfig:
        """Convert dictionary to configuration object."""
        robot_data = data.get('robot', {})
        robot_config = RobotConfig(**robot_data)
        
        solver_data = data.get('solver', {})
        solver_config = SolverConfig(**solver_data)
        
        evaluation_data = data.get('evaluation', {})
        evaluation_config = EvaluationConfig(**evaluation_data)
        
        simulation_data = data.get('simulation', {})
        simulation_config = SimulationConfig(**simulation_data)
        
        visualization_data = data.get('visualization', {})
        visualization_config = VisualizationConfig(**visualization_data)
        
        # Global settings
        global_data = {k: v for k, v in data.items() 
                      if k not in ['robot', 'solver', 'evaluation', 'simulation', 'visualization']}
        
        return IKConfig(
            robot=robot_config,
            solver=solver_config,
            evaluation=evaluation_config,
            simulation=simulation_config,
            visualization=visualization_config,
            **global_data
        )
    
    def _config_to_dict(self, config: IKConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        return {
            'robot': {
                'name': config.robot.name,
                'type': config.robot.type,
                'link_lengths': config.robot.link_lengths,
                'joint_limits': config.robot.joint_limits,
                'dh_params': config.robot.dh_params,
                'base_position': config.robot.base_position
            },
            'solver': {
                'name': config.solver.name,
                'type': config.solver.type,
                'tolerance': config.solver.tolerance,
                'max_iterations': config.solver.max_iterations,
                'damping': config.solver.damping,
                'step_size': config.solver.step_size,
                'initial_lambda': config.solver.initial_lambda,
                'lambda_multiplier': config.solver.lambda_multiplier,
                'null_space_gain': config.solver.null_space_gain
            },
            'evaluation': {
                'n_test_points': config.evaluation.n_test_points,
                'test_distributions': config.evaluation.test_distributions,
                'workspace_bounds': config.evaluation.workspace_bounds,
                'performance_metrics': config.evaluation.performance_metrics
            },
            'simulation': {
                'gui': config.simulation.gui,
                'timestep': config.simulation.timestep,
                'physics_engine': config.simulation.physics_engine,
                'ground_plane': config.simulation.ground_plane,
                'gravity': config.simulation.gravity
            },
            'visualization': {
                'backend': config.visualization.backend,
                'theme': config.visualization.theme,
                'figure_size': config.visualization.figure_size,
                'dpi': config.visualization.dpi,
                'save_plots': config.visualization.save_plots,
                'plot_format': config.visualization.plot_format
            },
            'random_seed': config.random_seed,
            'log_level': config.log_level,
            'output_dir': config.output_dir,
            'cache_dir': config.cache_dir
        }


# Predefined configurations
def get_planar_2r_config() -> IKConfig:
    """Get configuration for 2R planar robot with analytical solver."""
    robot_config = RobotConfig(
        name='planar_2r',
        type='planar_2r',
        link_lengths=(1.0, 1.0),
        joint_limits=[(-np.pi, np.pi), (-np.pi, np.pi)]
    )
    
    solver_config = SolverConfig(
        name='analytical_2r',
        type='analytical_2r',
        tolerance=1e-6
    )
    
    return IKConfig(robot=robot_config, solver=solver_config)


def get_planar_3r_config() -> IKConfig:
    """Get configuration for 3R planar robot with redundant solver."""
    robot_config = RobotConfig(
        name='planar_3r',
        type='planar_3r',
        link_lengths=(1.0, 1.0, 1.0),
        joint_limits=[(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    )
    
    solver_config = SolverConfig(
        name='redundant_ik',
        type='redundant',
        tolerance=1e-6,
        max_iterations=100,
        damping=0.01,
        null_space_gain=0.1
    )
    
    return IKConfig(robot=robot_config, solver=solver_config)


def get_spatial_6r_config() -> IKConfig:
    """Get configuration for 6R spatial robot with Newton-Raphson solver."""
    robot_config = RobotConfig(
        name='spatial_6r',
        type='spatial_6r',
        joint_limits=[(-np.pi, np.pi)] * 6
    )
    
    solver_config = SolverConfig(
        name='newton_raphson',
        type='newton_raphson',
        tolerance=1e-6,
        max_iterations=100,
        damping=0.01
    )
    
    return IKConfig(robot=robot_config, solver=solver_config)


def get_benchmark_config() -> IKConfig:
    """Get configuration for benchmarking multiple solvers."""
    robot_config = RobotConfig(
        name='planar_2r_benchmark',
        type='planar_2r',
        link_lengths=(1.0, 1.0)
    )
    
    solver_config = SolverConfig(
        name='benchmark_solver',
        type='newton_raphson',
        tolerance=1e-6,
        max_iterations=100
    )
    
    evaluation_config = EvaluationConfig(
        n_test_points=1000,
        test_distributions=['uniform', 'gaussian', 'edge_case']
    )
    
    return IKConfig(
        robot=robot_config,
        solver=solver_config,
        evaluation=evaluation_config
    )
