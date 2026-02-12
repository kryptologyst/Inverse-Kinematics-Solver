# Inverse Kinematics Solver

A comprehensive inverse kinematics solver for robotic manipulators with support for multiple robot models, various IK algorithms, simulation integration, and interactive visualization.

## Features

- **Multiple Robot Models**: 2R planar, 3R planar, and 6R spatial manipulators
- **Various IK Solvers**: Analytical, Newton-Raphson, Levenberg-Marquardt, Jacobian Transpose, and Redundant IK methods
- **Simulation Integration**: PyBullet physics simulation for testing and visualization
- **Comprehensive Evaluation**: Performance benchmarking with multiple metrics
- **Interactive Visualization**: Streamlit-based web interface for real-time testing
- **Modern Architecture**: Clean, typed code with proper error handling and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Inverse-Kinematics-Solver.git
cd Inverse-Kinematics-Solver

# Install dependencies
pip install -r requirements.txt

# Run basic examples
python examples/basic_examples.py

# Launch interactive visualization
streamlit run src/visualization/app.py
```

### Basic Usage

```python
from src.kinematics import create_ik_solver
import numpy as np

# Create a 2R planar robot with analytical solver
solver = create_ik_solver("planar_2r", "analytical_2r", link_lengths=(1.0, 1.0))

# Define target position
target_pose = np.array([1.5, 1.0])

# Solve IK
result = solver.solve(target_pose)

if result.success:
    print(f"Joint angles: {np.degrees(result.joint_angles):.2f}°")
    print(f"Error: {result.error:.2e}")
else:
    print(f"Failed: {result.message}")
```

## Robot Models

### Planar 2R Robot
- 2-DOF planar manipulator with two revolute joints
- Analytical IK solution available
- Workspace: annular region with inner radius |l1-l2| and outer radius l1+l2

### Planar 3R Robot
- 3-DOF planar manipulator with redundancy
- Requires numerical IK methods
- Enhanced workspace coverage and obstacle avoidance

### Spatial 6R Robot
- 6-DOF spatial manipulator following DH convention
- Full 3D workspace
- Supports position and orientation control

## IK Solvers

### Analytical 2R Solver
- Closed-form solution for 2-DOF planar arms
- Extremely fast (sub-millisecond)
- Provides both elbow-up and elbow-down solutions
- Only works with 2R planar robots

### Newton-Raphson Method
- Iterative method using Jacobian pseudo-inverse
- Fast convergence for well-conditioned problems
- Damped least squares for numerical stability
- Works with any robot model

### Levenberg-Marquardt Method
- Robust iterative method with adaptive damping
- Combines gradient descent and Gauss-Newton
- Good convergence properties
- Handles ill-conditioned problems well

### Jacobian Transpose Method
- Simple gradient-based approach
- Uses transpose instead of pseudo-inverse
- Slower convergence but more stable
- Good for learning and educational purposes

### Redundant IK Solver
- Handles redundant manipulators
- Uses null-space optimization
- Supports secondary objectives
- Maintains primary task while optimizing secondary goals

## Evaluation Metrics

- **Success Rate**: Percentage of successful IK solutions
- **Average Error**: Mean positioning error
- **Average Time**: Mean computation time
- **Workspace Coverage**: Percentage of reachable workspace
- **Robustness Score**: Consistency of error across different targets
- **Iteration Count**: Number of iterations to convergence

## Simulation Integration

The system integrates with PyBullet for physics simulation:

```python
from src.simulation.pybullet_sim import PyBulletSimulator, IKSimulationTester

# Create simulator
simulator = PyBulletSimulator(gui=True)

# Add robot
robot_id = simulator.add_planar_2r_robot("robot", link_lengths=(1.0, 1.0))

# Test IK solution
tester = IKSimulationTester(simulator)
result, sim_error = tester.test_ik_solution(solver, target_pose, "robot")
```

## Interactive Visualization

Launch the Streamlit web interface:

```bash
streamlit run src/visualization/app.py
```

Features:
- Interactive robot configuration
- Real-time IK solving
- Trajectory planning and visualization
- Solver benchmarking
- Performance analysis

## Configuration

The system uses YAML-based configuration:

```yaml
robot:
  name: planar_2r
  type: planar_2r
  link_lengths: [1.0, 1.0]
  joint_limits: [[-3.14159, 3.14159], [-3.14159, 3.14159]]

solver:
  name: analytical_2r
  type: analytical_2r
  tolerance: 1e-6
  max_iterations: 100

evaluation:
  n_test_points: 1000
  test_distributions: [uniform, gaussian, edge_case]
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/ -v
```

Tests include:
- Unit tests for all components
- Integration tests for IK solvers
- Performance benchmarks
- Simulation validation

## Examples

### Basic IK Solving
```python
# Solve for a single target
result = solver.solve(target_pose)
```

### Trajectory Planning
```python
# Solve for a trajectory
trajectory = np.array([[1.5, 1.0], [1.0, 1.5], [0.5, 0.5]])
results = solver.solve_trajectory(trajectory)
```

### Solver Comparison
```python
from src.evaluation.metrics import IKEvaluator

evaluator = IKEvaluator(robot_model)
test_targets = evaluator.generate_test_targets(1000, 'uniform')
benchmark_results = evaluator.compare_solvers(solvers, test_targets)
```

## Safety Notice

**IMPORTANT**: This software is for educational and research purposes only. Do not use on real hardware without proper safety measures and expert review. Always implement appropriate safety guards, velocity limits, and emergency stop mechanisms when working with real robots.

## Dependencies

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- Plotly
- Streamlit
- PyBullet (optional, for simulation)
- PyYAML
- Pytest (for testing)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{inverse_kinematics_solver,
  title={Inverse Kinematics Solver},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Inverse-Kinematics-Solver}
}
```
# Inverse-Kinematics-Solver
