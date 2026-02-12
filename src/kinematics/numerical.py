"""
Numerical inverse kinematics solvers.

This module implements various numerical methods for solving inverse kinematics,
including Newton-Raphson, Levenberg-Marquardt, and Jacobian transpose methods.
"""

import numpy as np
from typing import Optional, List
from .base import RobotModel, IKSolver, IKResult


class NewtonRaphsonIK(IKSolver):
    """
    Newton-Raphson method for inverse kinematics.
    
    Uses the Jacobian matrix to iteratively solve for joint angles that
    minimize the error between desired and actual end-effector pose.
    """
    
    def __init__(self, robot_model: RobotModel, tolerance: float = 1e-6, 
                 max_iterations: int = 100, damping: float = 0.01):
        """
        Initialize Newton-Raphson IK solver.
        
        Args:
            robot_model: Robot kinematic model
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            damping: Damping factor for numerical stability
        """
        super().__init__(robot_model, tolerance, max_iterations)
        self.damping = damping
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics using Newton-Raphson method.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            
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
        
        # Ensure initial guess is within joint limits
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
                    method="NewtonRaphson",
                    computation_time=computation_time,
                    message=f"Converged in {iteration + 1} iterations"
                )
            
            # Compute Jacobian
            J = self.robot_model.jacobian(joint_angles)
            
            # Compute pseudo-inverse with damping
            J_pinv = self._damped_pseudo_inverse(J)
            
            # Update joint angles
            delta_q = J_pinv @ error
            joint_angles += delta_q
            
            # Apply joint limits
            joint_angles = self.robot_model.clamp_joint_limits(joint_angles)
        
        # If we get here, we didn't converge
        final_pose = self.robot_model.forward_kinematics(joint_angles)
        final_error = np.linalg.norm(target_pose - final_pose)
        computation_time = time.time() - start_time
        
        return IKResult(
            success=False,
            joint_angles=joint_angles,
            error=final_error,
            iterations=self.max_iterations,
            method="NewtonRaphson",
            computation_time=computation_time,
            message=f"Failed to converge after {self.max_iterations} iterations"
        )
    
    def _damped_pseudo_inverse(self, J: np.ndarray) -> np.ndarray:
        """
        Compute damped pseudo-inverse of Jacobian.
        
        Args:
            J: Jacobian matrix
            
        Returns:
            Damped pseudo-inverse of Jacobian
        """
        # Damped least squares: J^T * (J * J^T + λ²I)^(-1)
        JJT = J @ J.T
        damping_matrix = self.damping**2 * np.eye(JJT.shape[0])
        return J.T @ np.linalg.inv(JJT + damping_matrix)


class LevenbergMarquardtIK(IKSolver):
    """
    Levenberg-Marquardt method for inverse kinematics.
    
    Combines gradient descent and Gauss-Newton methods with adaptive damping
    for robust convergence.
    """
    
    def __init__(self, robot_model: RobotModel, tolerance: float = 1e-6,
                 max_iterations: int = 100, initial_lambda: float = 0.01,
                 lambda_multiplier: float = 10.0):
        """
        Initialize Levenberg-Marquardt IK solver.
        
        Args:
            robot_model: Robot kinematic model
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            initial_lambda: Initial damping parameter
            lambda_multiplier: Lambda adjustment factor
        """
        super().__init__(robot_model, tolerance, max_iterations)
        self.lambda_val = initial_lambda
        self.lambda_multiplier = lambda_multiplier
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics using Levenberg-Marquardt method.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            
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
        
        # Compute initial error
        current_pose = self.robot_model.forward_kinematics(joint_angles)
        error = target_pose - current_pose
        error_norm = np.linalg.norm(error)
        prev_error_norm = error_norm
        
        for iteration in range(self.max_iterations):
            # Check convergence
            if error_norm < self.tolerance:
                computation_time = time.time() - start_time
                return IKResult(
                    success=True,
                    joint_angles=joint_angles,
                    error=error_norm,
                    iterations=iteration + 1,
                    method="LevenbergMarquardt",
                    computation_time=computation_time,
                    message=f"Converged in {iteration + 1} iterations"
                )
            
            # Compute Jacobian
            J = self.robot_model.jacobian(joint_angles)
            
            # Compute update direction
            delta_q = self._lm_update(J, error)
            
            # Try the update
            new_joint_angles = joint_angles + delta_q
            new_joint_angles = self.robot_model.clamp_joint_limits(new_joint_angles)
            
            # Compute new error
            new_pose = self.robot_model.forward_kinematics(new_joint_angles)
            new_error = target_pose - new_pose
            new_error_norm = np.linalg.norm(new_error)
            
            # Check if update improved the solution
            if new_error_norm < error_norm:
                # Accept the update
                joint_angles = new_joint_angles
                error_norm = new_error_norm
                error = new_error
                # Decrease lambda (more Gauss-Newton-like)
                self.lambda_val /= self.lambda_multiplier
            else:
                # Reject the update, increase lambda (more gradient descent-like)
                self.lambda_val *= self.lambda_multiplier
            
            prev_error_norm = error_norm
        
        # Final check
        computation_time = time.time() - start_time
        return IKResult(
            success=error_norm < self.tolerance,
            joint_angles=joint_angles,
            error=error_norm,
            iterations=self.max_iterations,
            method="LevenbergMarquardt",
            computation_time=computation_time,
            message=f"Final error: {error_norm:.2e}"
        )
    
    def _lm_update(self, J: np.ndarray, error: np.ndarray) -> np.ndarray:
        """
        Compute Levenberg-Marquardt update.
        
        Args:
            J: Jacobian matrix
            error: Current error vector
            
        Returns:
            Joint angle update
        """
        # (J^T * J + λI)^(-1) * J^T * error
        JTJ = J.T @ J
        damping_matrix = self.lambda_val * np.eye(JTJ.shape[0])
        return np.linalg.solve(JTJ + damping_matrix, J.T @ error)


class JacobianTransposeIK(IKSolver):
    """
    Jacobian transpose method for inverse kinematics.
    
    Uses the transpose of the Jacobian matrix instead of the pseudo-inverse.
    This method is simpler but typically slower to converge.
    """
    
    def __init__(self, robot_model: RobotModel, tolerance: float = 1e-6,
                 max_iterations: int = 1000, step_size: float = 0.1):
        """
        Initialize Jacobian transpose IK solver.
        
        Args:
            robot_model: Robot kinematic model
            tolerance: Convergence tolerance
            max_iterations: Maximum number of iterations
            step_size: Learning rate for gradient descent
        """
        super().__init__(robot_model, tolerance, max_iterations)
        self.step_size = step_size
    
    def solve(self, target_pose: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> IKResult:
        """
        Solve inverse kinematics using Jacobian transpose method.
        
        Args:
            target_pose: Desired end-effector pose
            initial_guess: Initial joint angle guess
            
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
                    method="JacobianTranspose",
                    computation_time=computation_time,
                    message=f"Converged in {iteration + 1} iterations"
                )
            
            # Compute Jacobian
            J = self.robot_model.jacobian(joint_angles)
            
            # Update using Jacobian transpose
            delta_q = self.step_size * J.T @ error
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
            method="JacobianTranspose",
            computation_time=computation_time,
            message=f"Failed to converge after {self.max_iterations} iterations"
        )
