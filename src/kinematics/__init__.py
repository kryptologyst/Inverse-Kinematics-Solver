"""
Main inverse kinematics solver interface and factory.

This module provides a unified interface for all IK solvers and includes
factory methods for creating different types of robots and solvers.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .base import RobotModel, IKSolver, IKResult
from .planar_2r import Planar2R, Analytical2RIK
from .numerical import NewtonRaphsonIK, LevenbergMarquardtIK, JacobianTransposeIK
from .spatial import Planar3R, Spatial6R, RedundantIK


class IKSolverFactory:
    """Factory class for creating IK solvers and robot models."""
    
    @staticmethod
    def create_robot(robot_type: str, **kwargs) -> RobotModel:
        """
        Create a robot model.
        
        Args:
            robot_type: Type of robot ('planar_2r', 'planar_3r', 'spatial_6r')
            **kwargs: Robot-specific parameters
            
        Returns:
            Robot model instance
        """
        if robot_type.lower() == 'planar_2r':
            link_lengths = kwargs.get('link_lengths', (1.0, 1.0))
            joint_limits = kwargs.get('joint_limits', None)
            return Planar2R(link_lengths, joint_limits)
        
        elif robot_type.lower() == 'planar_3r':
            link_lengths = kwargs.get('link_lengths', (1.0, 1.0, 1.0))
            joint_limits = kwargs.get('joint_limits', None)
            return Planar3R(link_lengths, joint_limits)
        
        elif robot_type.lower() == 'spatial_6r':
            dh_params = kwargs.get('dh_params', None)
            joint_limits = kwargs.get('joint_limits', None)
            return Spatial6R(dh_params, joint_limits)
        
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
    
    @staticmethod
    def create_solver(solver_type: str, robot_model: RobotModel, **kwargs) -> IKSolver:
        """
        Create an IK solver.
        
        Args:
            solver_type: Type of solver ('analytical_2r', 'newton_raphson', 
                         'levenberg_marquardt', 'jacobian_transpose', 'redundant')
            robot_model: Robot model to use
            **kwargs: Solver-specific parameters
            
        Returns:
            IK solver instance
        """
        if solver_type.lower() == 'analytical_2r':
            if not isinstance(robot_model, Planar2R):
                raise ValueError("Analytical2R solver requires Planar2R robot model")
            tolerance = kwargs.get('tolerance', 1e-6)
            return Analytical2RIK(robot_model, tolerance)
        
        elif solver_type.lower() == 'newton_raphson':
            tolerance = kwargs.get('tolerance', 1e-6)
            max_iterations = kwargs.get('max_iterations', 100)
            damping = kwargs.get('damping', 0.01)
            return NewtonRaphsonIK(robot_model, tolerance, max_iterations, damping)
        
        elif solver_type.lower() == 'levenberg_marquardt':
            tolerance = kwargs.get('tolerance', 1e-6)
            max_iterations = kwargs.get('max_iterations', 100)
            initial_lambda = kwargs.get('initial_lambda', 0.01)
            lambda_multiplier = kwargs.get('lambda_multiplier', 10.0)
            return LevenbergMarquardtIK(robot_model, tolerance, max_iterations, 
                                       initial_lambda, lambda_multiplier)
        
        elif solver_type.lower() == 'jacobian_transpose':
            tolerance = kwargs.get('tolerance', 1e-6)
            max_iterations = kwargs.get('max_iterations', 1000)
            step_size = kwargs.get('step_size', 0.1)
            return JacobianTransposeIK(robot_model, tolerance, max_iterations, step_size)
        
        elif solver_type.lower() == 'redundant':
            tolerance = kwargs.get('tolerance', 1e-6)
            max_iterations = kwargs.get('max_iterations', 100)
            damping = kwargs.get('damping', 0.01)
            null_space_gain = kwargs.get('null_space_gain', 0.1)
            return RedundantIK(robot_model, tolerance, max_iterations, damping, null_space_gain)
        
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")


class UnifiedIKSolver:
    """
    Unified interface for inverse kinematics solving.
    
    This class provides a single interface for solving inverse kinematics
    problems with different robot models and solver methods.
    """
    
    def __init__(self, robot_model: RobotModel, solver: IKSolver):
        """
        Initialize unified IK solver.
        
        Args:
            robot_model: Robot kinematic model
            solver: IK solver instance
        """
        self.robot_model = robot_model
        self.solver = solver
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None,
              **kwargs) -> IKResult:
        """
        Solve inverse kinematics.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            **kwargs: Additional solver-specific parameters
            
        Returns:
            IK solution result
        """
        return self.solver.solve(target_pose, initial_guess, **kwargs)
    
    def solve_multiple_targets(self, target_poses: List[np.ndarray],
                               initial_guess: Optional[np.ndarray] = None) -> List[IKResult]:
        """
        Solve IK for multiple target poses.
        
        Args:
            target_poses: List of desired end-effector poses
            initial_guess: Initial joint angle guess
            
        Returns:
            List of IK solution results
        """
        results = []
        current_guess = initial_guess
        
        for target_pose in target_poses:
            result = self.solve(target_pose, current_guess)
            results.append(result)
            
            # Use solution as initial guess for next target (if successful)
            if result.success:
                current_guess = result.joint_angles
        
        return results
    
    def solve_trajectory(self, target_trajectory: np.ndarray,
                         initial_guess: Optional[np.ndarray] = None) -> List[IKResult]:
        """
        Solve IK for a trajectory of target poses.
        
        Args:
            target_trajectory: Array of target poses (N x pose_dim)
            initial_guess: Initial joint angle guess
            
        Returns:
            List of IK solution results
        """
        target_poses = [target_trajectory[i] for i in range(len(target_trajectory))]
        return self.solve_multiple_targets(target_poses, initial_guess)
    
    def get_workspace_info(self) -> Dict[str, Union[float, Tuple[float, float]]]:
        """
        Get workspace information for the robot.
        
        Returns:
            Dictionary with workspace information
        """
        info = {
            'robot_name': self.robot_model.name,
            'n_joints': self.robot_model.n_joints,
            'joint_limits': self.robot_model.joint_limits
        }
        
        # Add robot-specific workspace info
        if hasattr(self.robot_model, 'workspace_radius'):
            min_radius, max_radius = self.robot_model.workspace_radius()
            info['workspace_radius'] = (min_radius, max_radius)
            info['workspace_area'] = np.pi * (max_radius**2 - min_radius**2)
        
        return info
    
    def validate_target(self, target_pose: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if a target pose is reachable.
        
        Args:
            target_pose: Target pose to validate
            
        Returns:
            (is_reachable, message)
        """
        # Basic validation
        if len(target_pose) != self.robot_model.n_joints:
            return False, f"Target pose dimension {len(target_pose)} doesn't match robot DOF {self.robot_model.n_joints}"
        
        # Robot-specific validation
        if hasattr(self.robot_model, 'workspace_radius'):
            if len(target_pose) >= 2:
                r = np.sqrt(target_pose[0]**2 + target_pose[1]**2)
                min_radius, max_radius = self.robot_model.workspace_radius()
                if r > max_radius or r < min_radius:
                    return False, f"Target distance {r:.3f} outside workspace [{min_radius:.3f}, {max_radius:.3f}]"
        
        return True, "Target appears reachable"


def create_ik_solver(robot_type: str, solver_type: str, **kwargs) -> UnifiedIKSolver:
    """
    Convenience function to create a complete IK solver setup.
    
    Args:
        robot_type: Type of robot ('planar_2r', 'planar_3r', 'spatial_6r')
        solver_type: Type of solver ('analytical_2r', 'newton_raphson', etc.)
        **kwargs: Parameters for robot and solver creation
        
    Returns:
        Unified IK solver instance
    """
    robot_model = IKSolverFactory.create_robot(robot_type, **kwargs)
    solver = IKSolverFactory.create_solver(solver_type, robot_model, **kwargs)
    return UnifiedIKSolver(robot_model, solver)
