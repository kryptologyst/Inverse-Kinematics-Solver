"""
Base classes and interfaces for inverse kinematics solvers.

This module provides the foundational classes and interfaces for implementing
various inverse kinematics algorithms for robotic manipulators.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class IKResult:
    """Result container for inverse kinematics solutions."""
    
    success: bool
    joint_angles: np.ndarray
    error: float
    iterations: int
    method: str
    computation_time: float
    message: str = ""


class RobotModel(ABC):
    """Abstract base class for robot kinematic models."""
    
    def __init__(self, name: str, joint_limits: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize robot model.
        
        Args:
            name: Robot model name
            joint_limits: List of (min, max) joint limits for each joint
        """
        self.name = name
        self.joint_limits = joint_limits
        self.n_joints = len(joint_limits) if joint_limits else 0
    
    @abstractmethod
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics.
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw] or [x, y] for 2D
        """
        pass
    
    @abstractmethod
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix.
        
        Args:
            joint_angles: Joint angles in radians
            
        Returns:
            Jacobian matrix
        """
        pass
    
    def check_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """
        Check if joint angles are within limits.
        
        Args:
            joint_angles: Joint angles to check
            
        Returns:
            True if all joints are within limits
        """
        if self.joint_limits is None:
            return True
        
        for i, (q_min, q_max) in enumerate(self.joint_limits):
            if not (q_min <= joint_angles[i] <= q_max):
                return False
        return True
    
    def clamp_joint_limits(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Clamp joint angles to limits.
        
        Args:
            joint_angles: Joint angles to clamp
            
        Returns:
            Clamped joint angles
        """
        if self.joint_limits is None:
            return joint_angles
        
        clamped = joint_angles.copy()
        for i, (q_min, q_max) in enumerate(self.joint_limits):
            clamped[i] = np.clip(clamped[i], q_min, q_max)
        return clamped


class IKSolver(ABC):
    """Abstract base class for inverse kinematics solvers."""
    
    def __init__(self, robot_model: RobotModel, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize IK solver.
        
        Args:
            robot_model: Robot kinematic model
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
        """
        self.robot_model = robot_model
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    @abstractmethod
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            
        Returns:
            IK solution result
        """
        pass
    
    def validate_target(self, target_pose: np.ndarray) -> bool:
        """
        Validate if target pose is reachable.
        
        Args:
            target_pose: Target pose to validate
            
        Returns:
            True if target appears reachable
        """
        # Basic validation - can be overridden by specific solvers
        return True
