"""
3D robotic arm models and IK solvers.

This module implements 3D robotic arm models including 3R and 6R manipulators
with their corresponding inverse kinematics solvers.
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import RobotModel, IKSolver, IKResult


class Planar3R(RobotModel):
    """
    3-DOF planar robotic arm (3 revolute joints).
    
    A planar manipulator with three revolute joints, all rotating around
    the z-axis. This provides redundancy in the planar workspace.
    """
    
    def __init__(self, link_lengths: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 joint_limits: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize 3R planar arm.
        
        Args:
            link_lengths: Lengths of the three links (l1, l2, l3)
            joint_limits: Joint limits [(min1, max1), (min2, max2), (min3, max3)]
        """
        self.link_lengths = np.array(link_lengths)
        self.l1, self.l2, self.l3 = self.link_lengths
        
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        
        super().__init__("Planar3R", joint_limits)
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for 3R planar arm.
        
        Args:
            joint_angles: Joint angles [theta1, theta2, theta3] in radians
            
        Returns:
            End-effector position [x, y]
        """
        theta1, theta2, theta3 = joint_angles
        
        # Compute cumulative angles
        angle1 = theta1
        angle2 = theta1 + theta2
        angle3 = theta1 + theta2 + theta3
        
        x = (self.l1 * np.cos(angle1) + 
             self.l2 * np.cos(angle2) + 
             self.l3 * np.cos(angle3))
        
        y = (self.l1 * np.sin(angle1) + 
             self.l2 * np.sin(angle2) + 
             self.l3 * np.sin(angle3))
        
        return np.array([x, y])
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for 3R planar arm.
        
        Args:
            joint_angles: Joint angles [theta1, theta2, theta3] in radians
            
        Returns:
            2x3 Jacobian matrix
        """
        theta1, theta2, theta3 = joint_angles
        
        # Compute cumulative angles
        angle1 = theta1
        angle2 = theta1 + theta2
        angle3 = theta1 + theta2 + theta3
        
        # Jacobian matrix for 3R planar arm
        J = np.array([
            [-self.l1 * np.sin(angle1) - self.l2 * np.sin(angle2) - self.l3 * np.sin(angle3),
             -self.l2 * np.sin(angle2) - self.l3 * np.sin(angle3),
             -self.l3 * np.sin(angle3)],
            [self.l1 * np.cos(angle1) + self.l2 * np.cos(angle2) + self.l3 * np.cos(angle3),
             self.l2 * np.cos(angle2) + self.l3 * np.cos(angle3),
             self.l3 * np.cos(angle3)]
        ])
        
        return J


class Spatial6R(RobotModel):
    """
    6-DOF spatial robotic arm (6 revolute joints).
    
    A general-purpose 6-DOF manipulator following the Denavit-Hartenberg
    convention. This is a simplified model with standard DH parameters.
    """
    
    def __init__(self, dh_params: Optional[List[Tuple[float, float, float, float]]] = None,
                 joint_limits: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize 6R spatial arm.
        
        Args:
            dh_params: DH parameters [(a, alpha, d, theta_offset), ...]
            joint_limits: Joint limits [(min1, max1), ..., (min6, max6)]
        """
        if dh_params is None:
            # Default DH parameters for a generic 6R arm
            dh_params = [
                (0.0, np.pi/2, 0.0, 0.0),      # Joint 1
                (0.4318, 0.0, 0.0, 0.0),       # Joint 2
                (0.0, -np.pi/2, 0.15005, 0.0), # Joint 3
                (0.0, np.pi/2, 0.4318, 0.0),   # Joint 4
                (0.0, -np.pi/2, 0.0, 0.0),     # Joint 5
                (0.0, 0.0, 0.0, 0.0)           # Joint 6
            ]
        
        self.dh_params = np.array(dh_params)
        
        if joint_limits is None:
            joint_limits = [(-np.pi, np.pi)] * 6
        
        super().__init__("Spatial6R", joint_limits)
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for 6R spatial arm.
        
        Args:
            joint_angles: Joint angles [theta1, ..., theta6] in radians
            
        Returns:
            End-effector pose [x, y, z, qx, qy, qz, qw]
        """
        T = np.eye(4)
        
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            
            # DH transformation matrix
            ct = np.cos(theta)
            st = np.sin(theta)
            ca = np.cos(alpha)
            sa = np.sin(alpha)
            
            T_i = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
        
        # Extract position and convert rotation matrix to quaternion
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        return np.concatenate([position, quaternion])
    
    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for 6R spatial arm.
        
        Args:
            joint_angles: Joint angles [theta1, ..., theta6] in radians
            
        Returns:
            6x6 Jacobian matrix (position and orientation)
        """
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated method
        h = 1e-6
        J = np.zeros((6, 6))
        
        current_pose = self.forward_kinematics(joint_angles)
        
        for i in range(6):
            joint_angles_pert = joint_angles.copy()
            joint_angles_pert[i] += h
            
            perturbed_pose = self.forward_kinematics(joint_angles_pert)
            J[:, i] = (perturbed_pose - current_pose) / h
        
        return J
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion (w, x, y, z).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [qx, qy, qz, qw]
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])


class RedundantIK(IKSolver):
    """
    Inverse kinematics solver for redundant manipulators.
    
    Uses the pseudo-inverse with null-space optimization to handle
    redundant degrees of freedom.
    """
    
    def __init__(self, robot_model: RobotModel, tolerance: float = 1e-6,
                 max_iterations: int = 100, damping: float = 0.01,
                 null_space_gain: float = 0.1):
        """
        Initialize redundant IK solver.
        
        Args:
            robot_model: Robot kinematic model
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            damping: Damping factor for numerical stability
            null_space_gain: Gain for null-space optimization
        """
        super().__init__(robot_model, tolerance, max_iterations)
        self.damping = damping
        self.null_space_gain = null_space_gain
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None,
              secondary_objective: Optional[callable] = None) -> IKResult:
        """
        Solve inverse kinematics with null-space optimization.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            secondary_objective: Optional secondary objective function
            
        Returns:
            IK solution result
        """
        import time
        start_time = time.time()
        
        # Initialize joint angles
        if initial_guess is None:
            joint_angles = np.zeros(self.robot_model.n_joints)
        else:
            joint_angles = initial_guess.copy()
        
        joint_angles = self.robot_model.clamp_joint_limits(joint_angles)
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector pose
            current_pose = self.robot_model.forward_kinematics(joint_angles)
            
            # Compute error
            error = target_pose - current_pose
            error_norm = np.linalg.norm(error)
            
            # Check convergence
            if error_norm < self.tolerance:
                computation_time = time.time() - start_time
                return IKResult(
                    success=True,
                    joint_angles=joint_angles,
                    error=error_norm,
                    iterations=iteration + 1,
                    method="RedundantIK",
                    computation_time=computation_time,
                    message=f"Converged in {iteration + 1} iterations"
                )
            
            # Compute Jacobian
            J = self.robot_model.jacobian(joint_angles)
            
            # Compute pseudo-inverse with damping
            J_pinv = self._damped_pseudo_inverse(J)
            
            # Primary task: minimize pose error
            delta_q_primary = J_pinv @ error
            
            # Secondary task: null-space optimization
            delta_q_secondary = np.zeros(self.robot_model.n_joints)
            if secondary_objective is not None:
                # Compute null-space projector
                I = np.eye(self.robot_model.n_joints)
                N = I - J_pinv @ J
                
                # Compute gradient of secondary objective
                h = 1e-6
                grad = np.zeros(self.robot_model.n_joints)
                for i in range(self.robot_model.n_joints):
                    joint_angles_pert = joint_angles.copy()
                    joint_angles_pert[i] += h
                    grad[i] = (secondary_objective(joint_angles_pert) - 
                              secondary_objective(joint_angles)) / h
                
                # Project gradient into null space
                delta_q_secondary = self.null_space_gain * N @ grad
            
            # Combine primary and secondary tasks
            delta_q = delta_q_primary + delta_q_secondary
            joint_angles += delta_q
            
            # Apply joint limits
            joint_angles = self.robot_model.clamp_joint_limits(joint_angles)
        
        # Final error
        final_pose = self.robot_model.forward_kinematics(joint_angles)
        final_error = np.linalg.norm(target_pose - final_pose)
        computation_time = time.time() - start_time
        
        return IKResult(
            success=False,
            joint_angles=joint_angles,
            error=final_error,
            iterations=self.max_iterations,
            method="RedundantIK",
            computation_time=computation_time,
            message=f"Failed to converge after {self.max_iterations} iterations"
        )
    
    def _damped_pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """Compute damped pseudo-inverse of Jacobian."""
        JJT = J @ J.T
        damping_matrix = self.damping**2 * np.eye(JJT.shape[0])
        return J.T @ np.linalg.inv(JJT + damping_matrix)
