"""
Main entry point and example scripts for the inverse kinematics solver.

This module provides example scripts demonstrating various features of the
IK solver system including basic usage, benchmarking, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any

from src.kinematics import create_ik_solver, UnifiedIKSolver
from src.evaluation.metrics import IKEvaluator
from src.simulation.pybullet_sim import PyBulletSimulator, IKSimulationTester
from src.config import get_planar_2r_config, get_benchmark_config


def basic_ik_example():
    """Basic example of using the IK solver."""
    print("Basic IK Solver Example")
    print("=" * 30)
    
    # Create a 2R planar robot with analytical solver
    solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
    
    # Define target position
    target_pose = np.array([1.5, 1.0])
    print(f"Target position: {target_pose}")
    
    # Solve IK
    result = solver.solve(target_pose)
    
    if result.success:
        print(f"✅ Success!")
        print(f"Joint angles: {np.degrees(result.joint_angles):.2f}°")
        print(f"Error: {result.error:.2e}")
        print(f"Computation time: {result.computation_time*1000:.2f} ms")
        
        # Verify solution
        actual_pose = solver.robot_model.forward_kinematics(result.joint_angles)
        print(f"Actual position: {actual_pose}")
        print(f"Position error: {np.linalg.norm(actual_pose - target_pose):.2e}")
    else:
        print(f"❌ Failed: {result.message}")


def trajectory_example():
    """Example of solving IK for a trajectory."""
    print("\nTrajectory IK Example")
    print("=" * 30)
    
    # Create solver
    solver = create_ik_solver("planar_2r", "newton_raphson", link_lengths=(1.0, 1.0))
    
    # Create circular trajectory
    angles = np.linspace(0, 2*np.pi, 20)
    radius = 1.2
    trajectory = np.array([
        [radius * np.cos(a), radius * np.sin(a)] for a in angles
    ])
    
    print(f"Solving trajectory with {len(trajectory)} points...")
    
    # Solve trajectory
    start_time = time.time()
    results = solver.solve_trajectory(trajectory)
    end_time = time.time()
    
    # Calculate statistics
    successful_results = [r for r in results if r.success]
    success_rate = len(successful_results) / len(results) * 100
    
    print(f"✅ Trajectory solved!")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {(end_time - start_time)*1000:.2f} ms")
    
    if successful_results:
        avg_error = np.mean([r.error for r in successful_results])
        avg_time = np.mean([r.computation_time for r in successful_results])
        print(f"Average error: {avg_error:.2e}")
        print(f"Average time per point: {avg_time*1000:.2f} ms")


def solver_comparison_example():
    """Example comparing different IK solvers."""
    print("\nSolver Comparison Example")
    print("=" * 30)
    
    # Create different solvers
    solvers = [
        create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0)),
        create_ik_solver("planar_2r", "newton_raphson", link_lengths=(1.0, 1.0)),
        create_ik_solver("planar_2r", "levenberg_marquardt", link_lengths=(1.0, 1.0)),
        create_ik_solver("planar_2r", "jacobian_transpose", link_lengths=(1.0, 1.0))
    ]
    
    # Test targets
    test_targets = [
        np.array([1.5, 1.0]),
        np.array([1.0, 1.5]),
        np.array([0.5, 0.5]),
        np.array([1.8, 0.2]),
        np.array([0.2, 1.8])
    ]
    
    print(f"Testing {len(solvers)} solvers on {len(test_targets)} targets...")
    
    results = {}
    
    for solver in solvers:
        solver_name = solver.solver.method
        solver_results = []
        
        for target in test_targets:
            result = solver.solve(target)
            solver_results.append(result)
        
        # Calculate statistics
        successful_results = [r for r in solver_results if r.success]
        success_rate = len(successful_results) / len(solver_results) * 100
        
        if successful_results:
            avg_error = np.mean([r.error for r in successful_results])
            avg_time = np.mean([r.computation_time for r in successful_results])
            avg_iterations = np.mean([r.iterations for r in successful_results])
        else:
            avg_error = float('inf')
            avg_time = float('inf')
            avg_iterations = float('inf')
        
        results[solver_name] = {
            'success_rate': success_rate,
            'avg_error': avg_error,
            'avg_time': avg_time,
            'avg_iterations': avg_iterations
        }
    
    # Print results
    print(f"\n{'Solver':<20} {'Success Rate':<12} {'Avg Error':<12} {'Avg Time (ms)':<15} {'Avg Iterations':<15}")
    print("-" * 80)
    
    for solver_name, stats in results.items():
        print(f"{solver_name:<20} {stats['success_rate']:<12.1f} {stats['avg_error']:<12.2e} "
              f"{stats['avg_time']*1000:<15.2f} {stats['avg_iterations']:<15.1f}")


def benchmarking_example():
    """Example of comprehensive benchmarking."""
    print("\nComprehensive Benchmarking Example")
    print("=" * 40)
    
    # Create evaluator
    robot_model = create_ik_solver("planar_2r", "analytical_2r").robot_model
    evaluator = IKEvaluator(robot_model)
    
    # Generate test targets
    test_targets = evaluator.generate_test_targets(500, 'uniform')
    print(f"Generated {len(test_targets)} test targets")
    
    # Create solvers to benchmark
    solvers = [
        create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0)),
        create_ik_solver("planar_2r", "newton_raphson", link_lengths=(1.0, 1.0)),
        create_ik_solver("planar_2r", "levenberg_marquardt", link_lengths=(1.0, 1.0))
    ]
    
    # Run benchmark
    print("Running benchmark...")
    benchmark_results = evaluator.compare_solvers(solvers, test_targets)
    
    # Generate report
    report = evaluator.generate_performance_report(benchmark_results)
    print("\n" + report)


def simulation_example():
    """Example using PyBullet simulation."""
    print("\nSimulation Example")
    print("=" * 20)
    
    # Create simulator
    simulator = PyBulletSimulator(gui=True)
    
    try:
        # Add robot
        robot_id = simulator.add_planar_2r_robot("test_robot", link_lengths=(1.0, 1.0))
        
        # Create IK solver
        solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
        
        # Create simulation tester
        tester = IKSimulationTester(simulator)
        
        # Test targets
        test_targets = [
            np.array([1.5, 1.0]),
            np.array([1.0, 1.5]),
            np.array([0.5, 0.5])
        ]
        
        print("Testing IK solutions in simulation...")
        
        for i, target in enumerate(test_targets):
            print(f"Testing target {i+1}: {target}")
            
            result, sim_error = tester.test_ik_solution(solver, target, "test_robot", visualize=True)
            
            if result.success:
                print(f"  ✅ IK Success: error={result.error:.2e}, time={result.computation_time*1000:.2f}ms")
                print(f"  ✅ Simulation error: {sim_error:.2e}")
            else:
                print(f"  ❌ IK Failed: {result.message}")
            
            # Small delay for visualization
            time.sleep(1)
        
        print("Simulation test completed!")
        
    finally:
        simulator.close()


def visualization_example():
    """Example of creating visualizations."""
    print("\nVisualization Example")
    print("=" * 25)
    
    # Create solver
    solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
    
    # Generate workspace visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot workspace boundary
    if hasattr(solver.robot_model, 'workspace_radius'):
        min_radius, max_radius = solver.robot_model.workspace_radius()
        
        # Outer boundary
        theta = np.linspace(0, 2*np.pi, 100)
        x_outer = max_radius * np.cos(theta)
        y_outer = max_radius * np.sin(theta)
        ax.plot(x_outer, y_outer, 'k--', alpha=0.5, label='Workspace Boundary')
        
        # Inner boundary
        if min_radius > 0:
            x_inner = min_radius * np.cos(theta)
            y_inner = min_radius * np.sin(theta)
            ax.plot(x_inner, y_inner, 'k--', alpha=0.5)
    
    # Test some target points
    test_targets = [
        np.array([1.5, 1.0]),
        np.array([1.0, 1.5]),
        np.array([0.5, 0.5]),
        np.array([1.8, 0.2])
    ]
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, target in enumerate(test_targets):
        result = solver.solve(target)
        
        if result.success:
            # Plot target
            ax.plot(target[0], target[1], 'o', color=colors[i], markersize=8, 
                   label=f'Target {i+1}')
            
            # Plot robot configuration
            joint_angles = result.joint_angles
            actual_pose = solver.robot_model.forward_kinematics(joint_angles)
            
            # Plot links (simplified for 2R)
            if hasattr(solver.robot_model, 'link_lengths') and len(solver.robot_model.link_lengths) == 2:
                l1, l2 = solver.robot_model.link_lengths
                theta1, theta2 = joint_angles
                
                # Joint positions
                joint1_pos = [0, 0]
                joint2_pos = [l1 * np.cos(theta1), l1 * np.sin(theta1)]
                end_pos = [actual_pose[0], actual_pose[1]]
                
                # Plot links
                ax.plot([joint1_pos[0], joint2_pos[0]], [joint1_pos[1], joint2_pos[1]], 
                       color=colors[i], linewidth=3, alpha=0.7)
                ax.plot([joint2_pos[0], end_pos[0]], [joint2_pos[1], end_pos[1]], 
                       color=colors[i], linewidth=3, alpha=0.7)
                
                # Plot joints
                ax.plot(joint1_pos[0], joint1_pos[1], 'ko', markersize=6)
                ax.plot(joint2_pos[0], joint2_pos[1], 'ko', markersize=6)
                ax.plot(end_pos[0], end_pos[1], 'ko', markersize=6)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Robot Workspace and IK Solutions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('workspace_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'workspace_visualization.png'")
    plt.show()


def main():
    """Run all examples."""
    print("Inverse Kinematics Solver Examples")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Run examples
        basic_ik_example()
        trajectory_example()
        solver_comparison_example()
        benchmarking_example()
        
        # Uncomment to run simulation example (requires PyBullet)
        # simulation_example()
        
        visualization_example()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    main()
