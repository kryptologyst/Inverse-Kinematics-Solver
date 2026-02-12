"""
2D robotic arm models and analytical IK solvers.

This module implements 2D robotic arm models with analytical inverse kinematics
solutions for simple planar manipulators.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import RobotModel, IKSolver, IKResult


class Planar2R(RobotModel):
    """
    2-DOF planar robotic arm (2 revolute joints).
    
    This is a simple 2D manipulator with two revolute joints and two links.
    The first joint rotates around the z-axis, and the second joint rotates
    around the z-axis at the end of the first link.
    """
    
    def __init__(self, link_lengths: Tuple[float, float] = (1.0, 1.0), 
                 joint_limits: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize 2R planar arm.
        
        Args:
            link_lengths: Lengths of the two links (l1, l2)
            joint_limits: Joint limits [(min1, max1), (min2, max2)]
        """
        self.link_lengths = np.array(link_lengths)
        self.l1, self.l2 = self.link_lengths
        
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi)]
        
        super().__init__("Planar2R", joint_limits)
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for 2R planar arm.
        
        Args:
            joint_angles: Joint angles [theta1, theta2] in radians
            
        Returns:
            End-effector position [x, y]
        """
        theta1, theta2 = joint_angles
        
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        
        return np.array([x, y])
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for 2R planar arm.
        
        Args:
            joint_angles: Joint angles [theta1, theta2] in radians
            
        Returns:
            2x2 Jacobian matrix
        """
        theta1, theta2 = joint_angles
        
        # Jacobian matrix for 2R planar arm
        J = np.array([
            [-self.l1 * np.sin(theta1) - self.l2 * np.sin(theta1 + theta2),
             -self.l2 * np.sin(theta1 + theta2)],
            [self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2),
             self.l2 * np.cos(theta1 + theta2)]
        ])
        
        return J
    
    def workspace_radius(self) -> Tuple[float, float]:
        """
        Compute workspace radius range.
        
        Returns:
            (min_radius, max_radius) of the workspace
        """
        min_radius = abs(self.l1 - self.l2)
        max_radius = self.l1 + self.l2
        return min_radius, max_radius


class Analytical2RIK(IKSolver):
    """
    Analytical inverse kinematics solver for 2R planar arm.
    
    Uses geometric approach with law of cosines to solve for joint angles
    given a target end-effector position.
    """
    
    def __init__(self, robot_model: Planar2R, tolerance: float = 1e-6):
        """
        Initialize analytical IK solver.
        
        Args:
            robot_model: 2R planar robot model
            tolerance: Convergence tolerance (not used in analytical solution)
        """
        super().__init__(robot_model, tolerance, max_iterations=1)
        self.robot_model = robot_model
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics analytically.
        
        Args:
            target_pose: Target position [x, y]
            initial_guess: Not used in analytical solution
            
        Returns:
            IK solution result
        """
        import time
        start_time = time.time()
        
        x, y = target_pose
        l1, l2 = self.robot_model.l1, self.robot_model.l2
        
        # Calculate distance to target
        r = np.sqrt(x**2 + y**2)
        
        # Check if target is reachable
        min_radius, max_radius = self.robot_model.workspace_radius()
        if r > max_radius or r < min_radius:
            return IKResult(
                success=False,
                joint_angles=np.zeros(2),
                error=float('inf'),
                iterations=0,
                method="Analytical2R",
                computation_time=time.time() - start_time,
                message=f"Target unreachable: r={r:.3f}, workspace=[{min_radius:.3f}, {max_radius:.3f}]"
            )
        
        # Use law of cosines to find theta2
        cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        
        # Handle numerical precision issues
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        
        # Two solutions: elbow up and elbow down
        theta2_up = np.arccos(cos_theta2)
        theta2_down = -theta2_up
        
        # Calculate theta1 for both solutions
        k1_up = l1 + l2 * np.cos(theta2_up)
        k2_up = l2 * np.sin(theta2_up)
        theta1_up = np.arctan2(y, x) - np.arctan2(k2_up, k1_up)
        
        k1_down = l1 + l2 * np.cos(theta2_down)
        k2_down = l2 * np.sin(theta2_down)
        theta1_down = np.arctan2(y, x) - np.arctan2(k2_down, k1_down)
        
        # Choose solution based on initial guess or default to elbow up
        if initial_guess is not None:
            # Choose solution closer to initial guess
            sol_up = np.array([theta1_up, theta2_up])
            sol_down = np.array([theta1_down, theta2_down])
            
            dist_up = np.linalg.norm(sol_up - initial_guess)
            dist_down = np.linalg.norm(sol_down - initial_guess)
            
            if dist_up < dist_down:
                theta1, theta2 = theta1_up, theta2_up
            else:
                theta1, theta2 = theta1_down, theta2_down
        else:
            # Default to elbow up configuration
            theta1, theta2 = theta1_up, theta2_up
        
        joint_angles = np.array([theta1, theta2])
        
        # Check joint limits
        if not self.robot_model.check_joint_limits(joint_angles):
            joint_angles = self.robot_model.clamp_joint_limits(joint_angles)
        
        # Verify solution
        actual_pose = self.robot_model.forward_kinematics(joint_angles)
        error = np.linalg.norm(actual_pose - target_pose)
        
        computation_time = time.time() - start_time
        
        return IKResult(
            success=True,
            joint_angles=joint_angles,
            error=error,
            iterations=1,
            method="Analytical2R",
            computation_time=computation_time,
            message="Analytical solution found"
        )
    
    def get_all_solutions(self, target_pose: np.ndarray) -> List[np.ndarray]:
        """
        Get all possible analytical solutions for a target pose.
        
        Args:
            target_pose: Target position [x, y]
            
        Returns:
            List of joint angle solutions
        """
        x, y = target_pose
        l1, l2 = self.robot_model.l1, self.robot_model.l2
        
        r = np.sqrt(x**2 + y**2)
        min_radius, max_radius = self.robot_model.workspace_radius()
        
        if r > max_radius or r < min_radius:
            return []
        
        cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        
        solutions = []
        
        # Elbow up solution
        theta2_up = np.arccos(cos_theta2)
        k1_up = l1 + l2 * np.cos(theta2_up)
        k2_up = l2 * np.sin(theta2_up)
        theta1_up = np.arctan2(y, x) - np.arctan2(k2_up, k1_up)
        solutions.append(np.array([theta1_up, theta2_up]))
        
        # Elbow down solution
        theta2_down = -theta2_up
        k1_down = l1 + l2 * np.cos(theta2_down)
        k2_down = l2 * np.sin(theta2_down)
        theta1_down = np.arctan2(y, x) - np.arctan2(k2_down, k1_down)
        solutions.append(np.array([theta1_down, theta2_down]))
        
        return solutions
