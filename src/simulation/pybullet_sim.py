"""
Simulation integration for inverse kinematics testing and visualization.

This module provides integration with PyBullet for physics simulation and
visualization of robotic manipulators and IK solutions.
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Optional, Tuple, Dict, Any
import time
import os
from .kinematics import UnifiedIKSolver, IKResult


class PyBulletSimulator:
    """
    PyBullet-based simulator for robotic manipulators.
    
    Provides physics simulation and visualization capabilities for testing
    inverse kinematics solutions.
    """
    
    def __init__(self, gui: bool = True, timestep: float = 1/240):
        """
        Initialize PyBullet simulator.
        
        Args:
            gui: Whether to show GUI
            timestep: Simulation timestep
        """
        self.gui = gui
        self.timestep = timestep
        
        # Initialize PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(timestep)
        
        # Ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Robot and objects storage
        self.robots: Dict[str, int] = {}
        self.objects: Dict[str, int] = {}
        self.target_markers: List[int] = []
    
    def add_planar_2r_robot(self, name: str, link_lengths: Tuple[float, float] = (1.0, 1.0),
                           base_position: Tuple[float, float, float] = (0, 0, 0)) -> int:
        """
        Add a 2R planar robot to the simulation.
        
        Args:
            name: Robot name
            link_lengths: Lengths of the two links
            base_position: Base position (x, y, z)
            
        Returns:
            Robot body ID
        """
        # Create simple 2R robot using basic shapes
        l1, l2 = link_lengths
        
        # Create robot links
        link1_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l1/2, 0.05, 0.05])
        link1_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[l1/2, 0.05, 0.05], 
                                          rgbaColor=[0.8, 0.2, 0.2, 1])
        
        link2_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l2/2, 0.05, 0.05])
        link2_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[l2/2, 0.05, 0.05], 
                                          rgbaColor=[0.2, 0.8, 0.2, 1])
        
        # Create multi-body robot
        robot_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=base_position,
            baseOrientation=[0, 0, 0, 1],
            linkMasses=[1.0, 1.0],
            linkCollisionShapeIndices=[link1_shape, link2_shape],
            linkVisualShapeIndices=[link1_visual, link2_visual],
            linkPositions=[[l1/2, 0, 0], [l1 + l2/2, 0, 0]],
            linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 1],
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]]
        )
        
        self.robots[name] = robot_id
        return robot_id
    
    def add_target_marker(self, position: Tuple[float, float, float], 
                          color: Tuple[float, float, float, float] = (1, 1, 0, 1)) -> int:
        """
        Add a target marker to the simulation.
        
        Args:
            position: Target position (x, y, z)
            color: Marker color (r, g, b, a)
            
        Returns:
            Marker body ID
        """
        marker_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=color)
        
        marker_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=marker_shape,
            baseVisualShapeIndex=marker_visual,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1]
        )
        
        self.target_markers.append(marker_id)
        return marker_id
    
    def set_robot_joints(self, robot_name: str, joint_angles: np.ndarray) -> None:
        """
        Set robot joint angles.
        
        Args:
            robot_name: Name of the robot
            joint_angles: Joint angles in radians
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot_id = self.robots[robot_name]
        
        for i, angle in enumerate(joint_angles):
            p.resetJointState(robot_id, i, angle)
    
    def get_robot_end_effector_position(self, robot_name: str) -> np.ndarray:
        """
        Get robot end-effector position.
        
        Args:
            robot_name: Name of the robot
            
        Returns:
            End-effector position [x, y, z]
        """
        if robot_name not in self.robots:
            raise ValueError(f"Robot '{robot_name}' not found")
        
        robot_id = self.robots[robot_name]
        
        # Get the last link state (end-effector)
        link_state = p.getLinkState(robot_id, len(self.robots[robot_name]) - 1)
        return np.array(link_state[0])  # Position
    
    def step_simulation(self) -> None:
        """Step the simulation forward."""
        p.stepSimulation()
    
    def run_simulation(self, duration: float) -> None:
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation duration in seconds
        """
        steps = int(duration / self.timestep)
        for _ in range(steps):
            self.step_simulation()
            if self.gui:
                time.sleep(self.timestep)
    
    def clear_target_markers(self) -> None:
        """Remove all target markers."""
        for marker_id in self.target_markers:
            p.removeBody(marker_id)
        self.target_markers.clear()
    
    def close(self) -> None:
        """Close the simulation."""
        p.disconnect(self.client)


class IKSimulationTester:
    """
    Test inverse kinematics solutions in simulation.
    
    Provides methods for testing IK solutions and visualizing results
    in PyBullet simulation.
    """
    
    def __init__(self, simulator: PyBulletSimulator):
        """
        Initialize IK simulation tester.
        
        Args:
            simulator: PyBullet simulator instance
        """
        self.simulator = simulator
    
    def test_ik_solution(self, ik_solver: UnifiedIKSolver, target_pose: np.ndarray,
                        robot_name: str, initial_guess: Optional[np.ndarray] = None,
                        visualize: bool = True) -> Tuple[IKResult, float]:
        """
        Test an IK solution in simulation.
        
        Args:
            ik_solver: IK solver to test
            target_pose: Target end-effector pose
            robot_name: Name of robot in simulation
            initial_guess: Initial joint angle guess
            visualize: Whether to add target marker
            
        Returns:
            (IK result, simulation error)
        """
        # Solve IK
        result = ik_solver.solve(target_pose, initial_guess)
        
        if not result.success:
            return result, float('inf')
        
        # Set robot joints
        self.simulator.set_robot_joints(robot_name, result.joint_angles)
        
        # Step simulation
        self.simulator.step_simulation()
        
        # Get actual end-effector position
        actual_position = self.simulator.get_robot_end_effector_position(robot_name)
        
        # Calculate simulation error
        if len(target_pose) >= 2:
            target_2d = target_pose[:2]
            actual_2d = actual_position[:2]
            simulation_error = np.linalg.norm(target_2d - actual_2d)
        else:
            simulation_error = np.linalg.norm(target_pose - actual_position)
        
        # Add target marker if requested
        if visualize and len(target_pose) >= 2:
            target_3d = [target_pose[0], target_pose[1], 0.1]
            self.simulator.add_target_marker(target_3d)
        
        return result, simulation_error
    
    def test_trajectory(self, ik_solver: UnifiedIKSolver, target_trajectory: np.ndarray,
                       robot_name: str, initial_guess: Optional[np.ndarray] = None,
                       visualize: bool = True) -> List[Tuple[IKResult, float]]:
        """
        Test IK solutions for a trajectory.
        
        Args:
            ik_solver: IK solver to test
            target_trajectory: Array of target poses (N x pose_dim)
            robot_name: Name of robot in simulation
            initial_guess: Initial joint angle guess
            visualize: Whether to add target markers
            
        Returns:
            List of (IK result, simulation error) tuples
        """
        results = []
        current_guess = initial_guess
        
        for i, target_pose in enumerate(target_trajectory):
            result, sim_error = self.test_ik_solution(
                ik_solver, target_pose, robot_name, current_guess, visualize
            )
            results.append((result, sim_error))
            
            # Use solution as initial guess for next target
            if result.success:
                current_guess = result.joint_angles
            
            # Small delay for visualization
            if visualize and self.simulator.gui:
                time.sleep(0.1)
        
        return results
    
    def benchmark_solvers(self, solvers: List[UnifiedIKSolver], target_poses: List[np.ndarray],
                        robot_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark multiple IK solvers.
        
        Args:
            solvers: List of IK solvers to benchmark
            target_poses: List of target poses to test
            robot_name: Name of robot in simulation
            
        Returns:
            Benchmark results dictionary
        """
        benchmark_results = {}
        
        for solver in solvers:
            solver_name = solver.solver.method
            results = []
            
            for target_pose in target_poses:
                result, sim_error = self.test_ik_solution(
                    solver, target_pose, robot_name, visualize=False
                )
                results.append({
                    'ik_result': result,
                    'simulation_error': sim_error,
                    'success': result.success
                })
            
            # Calculate statistics
            successful_results = [r for r in results if r['success']]
            success_rate = len(successful_results) / len(results) if results else 0
            
            if successful_results:
                avg_error = np.mean([r['ik_result'].error for r in successful_results])
                avg_time = np.mean([r['ik_result'].computation_time for r in successful_results])
                avg_iterations = np.mean([r['ik_result'].iterations for r in successful_results])
                avg_sim_error = np.mean([r['simulation_error'] for r in successful_results])
            else:
                avg_error = float('inf')
                avg_time = float('inf')
                avg_iterations = float('inf')
                avg_sim_error = float('inf')
            
            benchmark_results[solver_name] = {
                'success_rate': success_rate,
                'average_error': avg_error,
                'average_time': avg_time,
                'average_iterations': avg_iterations,
                'average_simulation_error': avg_sim_error,
                'total_tests': len(results),
                'successful_tests': len(successful_results)
            }
        
        return benchmark_results
