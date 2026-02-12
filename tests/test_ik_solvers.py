"""
Test suite for inverse kinematics solvers.

This module contains comprehensive tests for all IK solver implementations
including unit tests, integration tests, and performance tests.
"""

import unittest
import numpy as np
import time
from typing import List, Tuple

from src.kinematics import (
    create_ik_solver, UnifiedIKSolver, Planar2R, Planar3R, Spatial6R,
    Analytical2RIK, NewtonRaphsonIK, LevenbergMarquardtIK, JacobianTransposeIK
)
from src.evaluation.metrics import IKEvaluator


class TestRobotModels(unittest.TestCase):
    """Test robot model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.planar_2r = Planar2R(link_lengths=(1.0, 1.0))
        self.planar_3r = Planar3R(link_lengths=(1.0, 1.0, 1.0))
    
    def test_planar_2r_forward_kinematics(self):
        """Test 2R planar arm forward kinematics."""
        # Test known configurations
        joint_angles = np.array([0.0, 0.0])
        pose = self.planar_2r.forward_kinematics(joint_angles)
        expected = np.array([2.0, 0.0])  # Both links extended
        np.testing.assert_allclose(pose, expected, atol=1e-10)
        
        joint_angles = np.array([np.pi/2, 0.0])
        pose = self.planar_2r.forward_kinematics(joint_angles)
        expected = np.array([0.0, 2.0])  # Vertical configuration
        np.testing.assert_allclose(pose, expected, atol=1e-10)
    
    def test_planar_2r_jacobian(self):
        """Test 2R planar arm Jacobian computation."""
        joint_angles = np.array([0.0, 0.0])
        J = self.planar_2r.jacobian(joint_angles)
        
        # Expected Jacobian for theta1=0, theta2=0
        expected_J = np.array([
            [0.0, 0.0],
            [2.0, 1.0]
        ])
        np.testing.assert_allclose(J, expected_J, atol=1e-10)
    
    def test_planar_2r_workspace_radius(self):
        """Test workspace radius calculation."""
        min_radius, max_radius = self.planar_2r.workspace_radius()
        self.assertEqual(min_radius, 0.0)  # l1 - l2 = 0
        self.assertEqual(max_radius, 2.0)  # l1 + l2 = 2
    
    def test_joint_limits(self):
        """Test joint limit checking and clamping."""
        # Test within limits
        joint_angles = np.array([0.5, -0.5])
        self.assertTrue(self.planar_2r.check_joint_limits(joint_angles))
        
        # Test outside limits
        joint_angles = np.array([2*np.pi, -2*np.pi])
        self.assertFalse(self.planar_2r.check_joint_limits(joint_angles))
        
        # Test clamping
        clamped = self.planar_2r.clamp_joint_limits(joint_angles)
        self.assertTrue(self.planar_2r.check_joint_limits(clamped))


class TestIKSolvers(unittest.TestCase):
    """Test IK solver implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot_model = Planar2R(link_lengths=(1.0, 1.0))
        self.target_pose = np.array([1.5, 1.0])
    
    def test_analytical_2r_solver(self):
        """Test analytical 2R IK solver."""
        solver = Analytical2RIK(self.robot_model)
        result = solver.solve(self.target_pose)
        
        self.assertTrue(result.success)
        self.assertLess(result.error, 1e-6)
        self.assertEqual(result.iterations, 1)
        
        # Verify solution
        actual_pose = self.robot_model.forward_kinematics(result.joint_angles)
        np.testing.assert_allclose(actual_pose, self.target_pose, atol=1e-6)
    
    def test_analytical_2r_unreachable_target(self):
        """Test analytical 2R solver with unreachable target."""
        solver = Analytical2RIK(self.robot_model)
        unreachable_target = np.array([3.0, 0.0])  # Outside workspace
        result = solver.solve(unreachable_target)
        
        self.assertFalse(result.success)
        self.assertEqual(result.iterations, 0)
    
    def test_analytical_2r_multiple_solutions(self):
        """Test analytical 2R solver multiple solutions."""
        solver = Analytical2RIK(self.robot_model)
        solutions = solver.get_all_solutions(self.target_pose)
        
        self.assertEqual(len(solutions), 2)  # Elbow up and elbow down
        
        # Both solutions should reach the target
        for solution in solutions:
            actual_pose = self.robot_model.forward_kinematics(solution)
            np.testing.assert_allclose(actual_pose, self.target_pose, atol=1e-6)
    
    def test_newton_raphson_solver(self):
        """Test Newton-Raphson IK solver."""
        solver = NewtonRaphsonIK(self.robot_model, tolerance=1e-6, max_iterations=100)
        result = solver.solve(self.target_pose)
        
        self.assertTrue(result.success)
        self.assertLess(result.error, 1e-6)
        self.assertLess(result.iterations, 100)
        
        # Verify solution
        actual_pose = self.robot_model.forward_kinematics(result.joint_angles)
        np.testing.assert_allclose(actual_pose, self.target_pose, atol=1e-6)
    
    def test_levenberg_marquardt_solver(self):
        """Test Levenberg-Marquardt IK solver."""
        solver = LevenbergMarquardtIK(self.robot_model, tolerance=1e-6, max_iterations=100)
        result = solver.solve(self.target_pose)
        
        self.assertTrue(result.success)
        self.assertLess(result.error, 1e-6)
        self.assertLess(result.iterations, 100)
    
    def test_jacobian_transpose_solver(self):
        """Test Jacobian transpose IK solver."""
        solver = JacobianTransposeIK(self.robot_model, tolerance=1e-6, max_iterations=1000)
        result = solver.solve(self.target_pose)
        
        self.assertTrue(result.success)
        self.assertLess(result.error, 1e-6)
        self.assertLess(result.iterations, 1000)


class TestUnifiedIKSolver(unittest.TestCase):
    """Test unified IK solver interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
    
    def test_single_target_solving(self):
        """Test solving for a single target."""
        target_pose = np.array([1.5, 1.0])
        result = self.solver.solve(target_pose)
        
        self.assertTrue(result.success)
        self.assertLess(result.error, 1e-6)
    
    def test_multiple_targets_solving(self):
        """Test solving for multiple targets."""
        target_poses = [
            np.array([1.5, 1.0]),
            np.array([1.0, 1.5]),
            np.array([0.5, 0.5])
        ]
        
        results = self.solver.solve_multiple_targets(target_poses)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result.success)
    
    def test_trajectory_solving(self):
        """Test solving for a trajectory."""
        # Create a simple circular trajectory
        angles = np.linspace(0, 2*np.pi, 10)
        trajectory = np.array([
            [1.5 * np.cos(a), 1.5 * np.sin(a)] for a in angles
        ])
        
        results = self.solver.solve_trajectory(trajectory)
        
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertTrue(result.success)
    
    def test_workspace_info(self):
        """Test workspace information retrieval."""
        info = self.solver.get_workspace_info()
        
        self.assertIn('robot_name', info)
        self.assertIn('n_joints', info)
        self.assertIn('workspace_radius', info)
        self.assertEqual(info['n_joints'], 2)
    
    def test_target_validation(self):
        """Test target validation."""
        # Valid target
        valid_target = np.array([1.0, 1.0])
        is_valid, message = self.solver.validate_target(valid_target)
        self.assertTrue(is_valid)
        
        # Invalid target (wrong dimension)
        invalid_target = np.array([1.0, 1.0, 1.0])
        is_valid, message = self.solver.validate_target(invalid_target)
        self.assertFalse(is_valid)


class TestIKEvaluator(unittest.TestCase):
    """Test IK evaluator and benchmarking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot_model = Planar2R(link_lengths=(1.0, 1.0))
        self.evaluator = IKEvaluator(self.robot_model)
    
    def test_generate_test_targets(self):
        """Test test target generation."""
        targets = self.evaluator.generate_test_targets(100, 'uniform')
        
        self.assertEqual(len(targets), 100)
        for target in targets:
            self.assertEqual(len(target), 2)
    
    def test_evaluate_solver(self):
        """Test solver evaluation."""
        solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
        test_targets = self.evaluator.generate_test_targets(50, 'uniform')
        
        result = self.evaluator.evaluate_solver(solver, test_targets)
        
        self.assertGreater(result.success_rate, 0.8)  # Should have high success rate
        self.assertLess(result.average_error, 1e-6)
        self.assertGreater(result.average_time, 0)
    
    def test_compare_solvers(self):
        """Test solver comparison."""
        solvers = [
            create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0)),
            create_ik_solver("planar_2r", "newton_raphson", link_lengths=(1.0, 1.0))
        ]
        test_targets = self.evaluator.generate_test_targets(50, 'uniform')
        
        results = self.evaluator.compare_solvers(solvers, test_targets)
        
        self.assertEqual(len(results), 2)
        self.assertIn('Analytical2R', results)
        self.assertIn('NewtonRaphson', results)


class TestPerformance(unittest.TestCase):
    """Performance tests for IK solvers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.robot_model = Planar2R(link_lengths=(1.0, 1.0))
        self.test_targets = [
            np.array([1.5, 1.0]),
            np.array([1.0, 1.5]),
            np.array([0.5, 0.5]),
            np.array([1.8, 0.2])
        ]
    
    def test_analytical_solver_performance(self):
        """Test analytical solver performance."""
        solver = Analytical2RIK(self.robot_model)
        
        start_time = time.time()
        for target in self.test_targets:
            result = solver.solve(target)
            self.assertTrue(result.success)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(self.test_targets)
        self.assertLess(avg_time, 0.001)  # Should be very fast
    
    def test_numerical_solver_performance(self):
        """Test numerical solver performance."""
        solver = NewtonRaphsonIK(self.robot_model, tolerance=1e-6, max_iterations=50)
        
        start_time = time.time()
        for target in self.test_targets:
            result = solver.solve(target)
            self.assertTrue(result.success)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(self.test_targets)
        self.assertLess(avg_time, 0.01)  # Should be reasonably fast


if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Run tests
    unittest.main(verbosity=2)
