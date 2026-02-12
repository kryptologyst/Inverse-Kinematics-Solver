#!/usr/bin/env python3
"""
Quick start script for the Inverse Kinematics Solver.

This script provides a simple way to get started with the IK solver
and demonstrates its basic functionality.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kinematics import create_ik_solver


def quick_demo():
    """Run a quick demonstration of the IK solver."""
    print("🤖 Inverse Kinematics Solver - Quick Demo")
    print("=" * 45)
    
    # Create a simple 2R planar robot
    print("Creating 2R planar robot with analytical solver...")
    solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))
    
    # Test some target positions
    targets = [
        np.array([1.5, 1.0]),
        np.array([1.0, 1.5]),
        np.array([0.5, 0.5]),
        np.array([1.8, 0.2])
    ]
    
    print(f"\nTesting {len(targets)} target positions:")
    print("-" * 30)
    
    for i, target in enumerate(targets, 1):
        print(f"Target {i}: ({target[0]:.1f}, {target[1]:.1f})")
        
        result = solver.solve(target)
        
        if result.success:
            print(f"  ✅ Success!")
            print(f"  Joint angles: {np.degrees(result.joint_angles):.1f}°")
            print(f"  Error: {result.error:.2e}")
            print(f"  Time: {result.computation_time*1000:.2f}ms")
        else:
            print(f"  ❌ Failed: {result.message}")
        print()
    
    # Show workspace info
    info = solver.get_workspace_info()
    print("Robot Information:")
    print(f"  Name: {info['robot_name']}")
    print(f"  Joints: {info['n_joints']}")
    if 'workspace_radius' in info:
        min_r, max_r = info['workspace_radius']
        print(f"  Workspace: {min_r:.1f}m to {max_r:.1f}m radius")
    
    print("\n🎉 Quick demo completed!")
    print("\nNext steps:")
    print("1. Run 'python examples/basic_examples.py' for more examples")
    print("2. Run 'streamlit run src/visualization/app.py' for interactive demo")
    print("3. Run 'python modernized_original.py' to see the modernized original")


if __name__ == "__main__":
    try:
        quick_demo()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your installation and try again.")
