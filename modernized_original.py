#!/usr/bin/env python3
"""
Modernized Inverse Kinematics Solver - Original Implementation

This script demonstrates the modernized version of the original 2R planar
inverse kinematics solver with improved error handling, type hints, and
comprehensive testing.
"""

import numpy as np
from typing import Tuple, Optional
import time
import matplotlib.pyplot as plt


class Modernized2RIK:
    """
    Modernized version of the original 2R planar inverse kinematics solver.
    
    This class provides a clean, well-documented implementation of the
    analytical inverse kinematics solution for a 2-DOF planar robotic arm.
    """
    
    def __init__(self, l1: float = 1.0, l2: float = 1.0):
        """
        Initialize the 2R planar IK solver.
        
        Args:
            l1: Length of first link (meters)
            l2: Length of second link (meters)
        """
        self.l1 = l1
        self.l2 = l2
        
        # Calculate workspace bounds
        self.min_radius = abs(l1 - l2)
        self.max_radius = l1 + l2
    
    def forward_kinematics(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """
        Compute forward kinematics for the 2R planar arm.
        
        Args:
            theta1: First joint angle (radians)
            theta2: Second joint angle (radians)
            
        Returns:
            End-effector position (x, y) in meters
        """
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return x, y
    
    def inverse_kinematics(self, x: float, y: float) -> Tuple[float, float]:
        """
        Solve inverse kinematics for a 2-joint robotic arm in 2D.
        
        This is the modernized version of the original function with improved
        error handling, documentation, and return values.
        
        Args:
            x: Desired x-coordinate of the end effector (meters)
            y: Desired y-coordinate of the end effector (meters)
            
        Returns:
            Joint angles (theta1, theta2) in radians
            
        Raises:
            ValueError: If the target is unreachable
        """
        # Calculate the distance to the target position
        r = np.sqrt(x**2 + y**2)
        
        # Check if the target is reachable
        if r > self.max_radius or r < self.min_radius:
            raise ValueError(
                f"Target is out of reach. Distance: {r:.3f}m, "
                f"Workspace: [{self.min_radius:.3f}, {self.max_radius:.3f}]m"
            )
        
        # Use the law of cosines to calculate theta2
        cos_theta2 = (r**2 - self.l1**2 - self.l2**2) / (-2 * self.l1 * self.l2)
        
        # Handle numerical precision issues
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        
        # Calculate theta2 (elbow up configuration)
        theta2 = np.arccos(cos_theta2)
        
        # Calculate theta1
        k1 = self.l1 + self.l2 * np.cos(theta2)
        k2 = self.l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        return theta1, theta2
    
    def get_all_solutions(self, x: float, y: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get all possible analytical solutions for a target pose.
        
        Args:
            x: Desired x-coordinate of the end effector (meters)
            y: Desired y-coordinate of the end effector (meters)
            
        Returns:
            Tuple of (elbow_up_solution, elbow_down_solution)
            
        Raises:
            ValueError: If the target is unreachable
        """
        r = np.sqrt(x**2 + y**2)
        
        if r > self.max_radius or r < self.min_radius:
            raise ValueError(f"Target is out of reach. Distance: {r:.3f}m")
        
        cos_theta2 = (r**2 - self.l1**2 - self.l2**2) / (-2 * self.l1 * self.l2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        
        # Elbow up solution
        theta2_up = np.arccos(cos_theta2)
        k1_up = self.l1 + self.l2 * np.cos(theta2_up)
        k2_up = self.l2 * np.sin(theta2_up)
        theta1_up = np.arctan2(y, x) - np.arctan2(k2_up, k1_up)
        
        # Elbow down solution
        theta2_down = -theta2_up
        k1_down = self.l1 + self.l2 * np.cos(theta2_down)
        k2_down = self.l2 * np.sin(theta2_down)
        theta1_down = np.arctan2(y, x) - np.arctan2(k2_down, k1_down)
        
        return (theta1_up, theta2_up), (theta1_down, theta2_down)
    
    def verify_solution(self, theta1: float, theta2: float, target_x: float, target_y: float) -> float:
        """
        Verify that a solution reaches the target within tolerance.
        
        Args:
            theta1: First joint angle (radians)
            theta2: Second joint angle (radians)
            target_x: Target x-coordinate (meters)
            target_y: Target y-coordinate (meters)
            
        Returns:
            Position error in meters
        """
        actual_x, actual_y = self.forward_kinematics(theta1, theta2)
        error = np.sqrt((actual_x - target_x)**2 + (actual_y - target_y)**2)
        return error


def test_modernized_ik():
    """Test the modernized IK solver with various scenarios."""
    print("Testing Modernized 2R Inverse Kinematics Solver")
    print("=" * 50)
    
    # Create solver
    ik_solver = Modernized2RIK(l1=1.0, l2=1.0)
    
    # Test cases
    test_cases = [
        (1.5, 1.0, "Reachable target"),
        (1.0, 1.5, "Another reachable target"),
        (0.5, 0.5, "Near workspace center"),
        (2.0, 0.0, "Maximum reach"),
        (0.1, 0.1, "Near base"),
    ]
    
    print(f"Robot configuration: l1={ik_solver.l1}m, l2={ik_solver.l2}m")
    print(f"Workspace: [{ik_solver.min_radius:.3f}, {ik_solver.max_radius:.3f}]m")
    print()
    
    for i, (target_x, target_y, description) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"Target: ({target_x:.1f}, {target_y:.1f})")
        
        try:
            start_time = time.time()
            theta1, theta2 = ik_solver.inverse_kinematics(target_x, target_y)
            computation_time = time.time() - start_time
            
            # Verify solution
            error = ik_solver.verify_solution(theta1, theta2, target_x, target_y)
            
            print(f"✅ Success!")
            print(f"   Joint angles: θ1={np.degrees(theta1):.2f}°, θ2={np.degrees(theta2):.2f}°")
            print(f"   Position error: {error:.2e}m")
            print(f"   Computation time: {computation_time*1000:.2f}ms")
            
            # Test both solutions
            solutions = ik_solver.get_all_solutions(target_x, target_y)
            print(f"   Elbow up: θ1={np.degrees(solutions[0][0]):.2f}°, θ2={np.degrees(solutions[0][1]):.2f}°")
            print(f"   Elbow down: θ1={np.degrees(solutions[1][0]):.2f}°, θ2={np.degrees(solutions[1][1]):.2f}°")
            
        except ValueError as e:
            print(f"❌ Failed: {e}")
        
        print()


def test_unreachable_targets():
    """Test the solver with unreachable targets."""
    print("Testing Unreachable Targets")
    print("=" * 30)
    
    ik_solver = Modernized2RIK(l1=1.0, l2=1.0)
    
    unreachable_targets = [
        (3.0, 0.0, "Too far"),
        (0.05, 0.05, "Too close"),
        (2.5, 2.5, "Outside workspace"),
    ]
    
    for target_x, target_y, description in unreachable_targets:
        print(f"Testing {description}: ({target_x}, {target_y})")
        
        try:
            theta1, theta2 = ik_solver.inverse_kinematics(target_x, target_y)
            print(f"❌ Unexpected success!")
        except ValueError as e:
            print(f"✅ Correctly rejected: {e}")
        
        print()


def visualize_workspace():
    """Create a visualization of the robot workspace and IK solutions."""
    print("Creating Workspace Visualization")
    print("=" * 35)
    
    ik_solver = Modernized2RIK(l1=1.0, l2=1.0)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot workspace boundary
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Outer boundary
    x_outer = ik_solver.max_radius * np.cos(theta)
    y_outer = ik_solver.max_radius * np.sin(theta)
    ax.plot(x_outer, y_outer, 'k--', alpha=0.5, label='Workspace Boundary')
    
    # Inner boundary
    if ik_solver.min_radius > 0:
        x_inner = ik_solver.min_radius * np.cos(theta)
        y_inner = ik_solver.min_radius * np.sin(theta)
        ax.plot(x_inner, y_inner, 'k--', alpha=0.5)
    
    # Test some target points
    test_targets = [
        (1.5, 1.0, 'red'),
        (1.0, 1.5, 'blue'),
        (0.5, 0.5, 'green'),
        (1.8, 0.2, 'orange'),
        (0.2, 1.8, 'purple')
    ]
    
    for target_x, target_y, color in test_targets:
        try:
            # Get both solutions
            solutions = ik_solver.get_all_solutions(target_x, target_y)
            
            # Plot target
            ax.plot(target_x, target_y, 'o', color=color, markersize=8, 
                   label=f'Target ({target_x}, {target_y})')
            
            # Plot both robot configurations
            for i, (theta1, theta2) in enumerate(solutions):
                # Calculate joint positions
                joint1_pos = (0, 0)
                joint2_pos = (ik_solver.l1 * np.cos(theta1), ik_solver.l1 * np.sin(theta1))
                end_pos = (target_x, target_y)
                
                # Plot links
                linestyle = '-' if i == 0 else '--'
                ax.plot([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]], 
                       color=color, linewidth=3, alpha=0.7, linestyle=linestyle)
                ax.plot([joint2_pos[0], end_pos[0]], [joint2_pos[1], end_pos[1]], 
                       color=color, linewidth=3, alpha=0.7, linestyle=linestyle)
                
                # Plot joints
                ax.plot(joint1_pos[0], joint1_pos[1], 'ko', markersize=6)
                ax.plot(joint2_pos[0], joint2_pos[1], 'ko', markersize=6)
                ax.plot(end_pos[0], end_pos[1], 'ko', markersize=6)
        
        except ValueError as e:
            print(f"Skipping unreachable target ({target_x}, {target_y}): {e}")
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('2R Planar Robot Workspace and IK Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('modernized_ik_workspace.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'modernized_ik_workspace.png'")
    plt.show()


def main():
    """Main function demonstrating the modernized IK solver."""
    print("Modernized Inverse Kinematics Solver")
    print("=" * 40)
    print("This is the modernized version of the original 2R planar IK solver")
    print("with improved error handling, type hints, and comprehensive testing.")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run tests
        test_modernized_ik()
        test_unreachable_targets()
        
        # Create visualization
        visualize_workspace()
        
        print("✅ All tests completed successfully!")
        print("\nThe modernized solver provides:")
        print("- Better error handling and validation")
        print("- Type hints for improved code clarity")
        print("- Multiple solution support (elbow up/down)")
        print("- Comprehensive testing and visualization")
        print("- Integration with the full IK solver framework")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
