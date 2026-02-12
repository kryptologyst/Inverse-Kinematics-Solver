"""
Interactive visualization and demo application.

This module provides a Streamlit-based web interface for interactive
inverse kinematics testing and visualization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import List, Dict, Any, Tuple
import pandas as pd

from ..kinematics import create_ik_solver, UnifiedIKSolver
from ..simulation.pybullet_sim import PyBulletSimulator, IKSimulationTester
from ..evaluation.metrics import IKEvaluator


class IKVisualizationApp:
    """
    Streamlit application for interactive IK visualization and testing.
    """
    
    def __init__(self):
        """Initialize the visualization app."""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="Inverse Kinematics Solver",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'robot_model' not in st.session_state:
            st.session_state.robot_model = None
        if 'ik_solver' not in st.session_state:
            st.session_state.ik_solver = None
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = []
        if 'benchmark_results' not in st.session_state:
            st.session_state.benchmark_results = {}
    
    def run(self):
        """Run the main application."""
        st.title("🤖 Inverse Kinematics Solver")
        st.markdown("Interactive testing and visualization of inverse kinematics algorithms")
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["Interactive Testing", "Trajectory Planning", "Benchmarking", "Documentation"])
        
        with tab1:
            self.render_interactive_testing()
        
        with tab2:
            self.render_trajectory_planning()
        
        with tab3:
            self.render_benchmarking()
        
        with tab4:
            self.render_documentation()
    
    def render_sidebar(self):
        """Render the sidebar configuration."""
        st.sidebar.header("Configuration")
        
        # Robot selection
        robot_type = st.sidebar.selectbox(
            "Robot Type",
            ["planar_2r", "planar_3r", "spatial_6r"],
            index=0
        )
        
        # Solver selection
        solver_type = st.sidebar.selectbox(
            "IK Solver",
            ["analytical_2r", "newton_raphson", "levenberg_marquardt", 
             "jacobian_transpose", "redundant"],
            index=0
        )
        
        # Robot parameters
        st.sidebar.subheader("Robot Parameters")
        
        if robot_type == "planar_2r":
            l1 = st.sidebar.slider("Link 1 Length", 0.1, 2.0, 1.0, 0.1)
            l2 = st.sidebar.slider("Link 2 Length", 0.1, 2.0, 1.0, 0.1)
            link_lengths = (l1, l2)
        elif robot_type == "planar_3r":
            l1 = st.sidebar.slider("Link 1 Length", 0.1, 2.0, 1.0, 0.1)
            l2 = st.sidebar.slider("Link 2 Length", 0.1, 2.0, 1.0, 0.1)
            l3 = st.sidebar.slider("Link 3 Length", 0.1, 2.0, 1.0, 0.1)
            link_lengths = (l1, l2, l3)
        else:
            link_lengths = None
        
        # Solver parameters
        st.sidebar.subheader("Solver Parameters")
        tolerance = st.sidebar.number_input("Tolerance", 1e-8, 1e-3, 1e-6, format="%.0e")
        max_iterations = st.sidebar.slider("Max Iterations", 10, 1000, 100)
        
        # Create solver
        if st.sidebar.button("Create Solver"):
            try:
                if link_lengths:
                    solver = create_ik_solver(robot_type, solver_type, 
                                            link_lengths=link_lengths,
                                            tolerance=tolerance,
                                            max_iterations=max_iterations)
                else:
                    solver = create_ik_solver(robot_type, solver_type,
                                            tolerance=tolerance,
                                            max_iterations=max_iterations)
                
                st.session_state.ik_solver = solver
                st.sidebar.success("Solver created successfully!")
            except Exception as e:
                st.sidebar.error(f"Error creating solver: {str(e)}")
    
    def render_interactive_testing(self):
        """Render the interactive testing tab."""
        st.header("Interactive Testing")
        
        if st.session_state.ik_solver is None:
            st.warning("Please create a solver in the sidebar first.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Target Configuration")
            
            # Target input
            target_x = st.number_input("Target X", -2.0, 2.0, 1.5, 0.1)
            target_y = st.number_input("Target Y", -2.0, 2.0, 1.0, 0.1)
            target_pose = np.array([target_x, target_y])
            
            # Initial guess
            st.subheader("Initial Guess (Optional)")
            use_initial_guess = st.checkbox("Use initial guess")
            initial_guess = None
            
            if use_initial_guess:
                n_joints = st.session_state.ik_solver.robot_model.n_joints
                initial_guess = np.zeros(n_joints)
                for i in range(n_joints):
                    initial_guess[i] = st.number_input(
                        f"Joint {i+1} (rad)", 
                        -np.pi, np.pi, 0.0, 0.1
                    )
            
            # Solve button
            if st.button("Solve IK", type="primary"):
                with st.spinner("Solving..."):
                    result = st.session_state.ik_solver.solve(target_pose, initial_guess)
                
                if result.success:
                    st.success("✅ IK solved successfully!")
                    st.write(f"**Joint Angles:** {np.degrees(result.joint_angles):.2f}°")
                    st.write(f"**Error:** {result.error:.2e}")
                    st.write(f"**Iterations:** {result.iterations}")
                    st.write(f"**Time:** {result.computation_time*1000:.2f} ms")
                else:
                    st.error(f"❌ IK solving failed: {result.message}")
        
        with col2:
            st.subheader("Visualization")
            
            if st.session_state.ik_solver and st.button("Show Robot Configuration"):
                self.plot_robot_configuration(target_pose, result.joint_angles if 'result' in locals() else None)
    
    def render_trajectory_planning(self):
        """Render the trajectory planning tab."""
        st.header("Trajectory Planning")
        
        if st.session_state.ik_solver is None:
            st.warning("Please create a solver in the sidebar first.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Trajectory Configuration")
            
            # Trajectory type
            traj_type = st.selectbox(
                "Trajectory Type",
                ["Circle", "Line", "Custom Points"]
            )
            
            if traj_type == "Circle":
                center_x = st.number_input("Center X", -2.0, 2.0, 0.0, 0.1)
                center_y = st.number_input("Center Y", -2.0, 2.0, 0.0, 0.1)
                radius = st.slider("Radius", 0.1, 1.5, 0.8, 0.1)
                n_points = st.slider("Number of Points", 10, 100, 50)
                
                angles = np.linspace(0, 2*np.pi, n_points)
                trajectory = np.array([
                    [center_x + radius * np.cos(a), center_y + radius * np.sin(a)]
                    for a in angles
                ])
            
            elif traj_type == "Line":
                start_x = st.number_input("Start X", -2.0, 2.0, -1.0, 0.1)
                start_y = st.number_input("Start Y", -2.0, 2.0, -1.0, 0.1)
                end_x = st.number_input("End X", -2.0, 2.0, 1.0, 0.1)
                end_y = st.number_input("End Y", -2.0, 2.0, 1.0, 0.1)
                n_points = st.slider("Number of Points", 10, 100, 50)
                
                trajectory = np.array([
                    np.linspace(start_x, end_x, n_points),
                    np.linspace(start_y, end_y, n_points)
                ]).T
            
            else:  # Custom Points
                st.write("Enter custom trajectory points:")
                n_points = st.slider("Number of Points", 2, 20, 5)
                trajectory = []
                
                for i in range(n_points):
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x = st.number_input(f"Point {i+1} X", -2.0, 2.0, 0.0, 0.1, key=f"x_{i}")
                    with col_y:
                        y = st.number_input(f"Point {i+1} Y", -2.0, 2.0, 0.0, 0.1, key=f"y_{i}")
                    trajectory.append([x, y])
                
                trajectory = np.array(trajectory)
            
            # Solve trajectory
            if st.button("Solve Trajectory", type="primary"):
                with st.spinner("Solving trajectory..."):
                    results = st.session_state.ik_solver.solve_trajectory(trajectory)
                
                # Calculate statistics
                successful_results = [r for r in results if r.success]
                success_rate = len(successful_results) / len(results) * 100
                
                st.success(f"✅ Trajectory solved! Success rate: {success_rate:.1f}%")
                
                if successful_results:
                    avg_error = np.mean([r.error for r in successful_results])
                    avg_time = np.mean([r.computation_time for r in successful_results])
                    st.write(f"**Average Error:** {avg_error:.2e}")
                    st.write(f"**Average Time:** {avg_time*1000:.2f} ms")
                
                st.session_state.simulation_results = results
        
        with col2:
            st.subheader("Trajectory Visualization")
            
            if st.session_state.simulation_results:
                self.plot_trajectory_results(trajectory, st.session_state.simulation_results)
    
    def render_benchmarking(self):
        """Render the benchmarking tab."""
        st.header("Solver Benchmarking")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Benchmark Configuration")
            
            # Test parameters
            n_tests = st.slider("Number of Test Points", 100, 1000, 500)
            test_distribution = st.selectbox(
                "Test Distribution",
                ["uniform", "gaussian", "edge_case"]
            )
            
            # Solver selection
            available_solvers = ["analytical_2r", "newton_raphson", "levenberg_marquardt", 
                               "jacobian_transpose", "redundant"]
            selected_solvers = st.multiselect(
                "Solvers to Benchmark",
                available_solvers,
                default=["analytical_2r", "newton_raphson"]
            )
            
            # Run benchmark
            if st.button("Run Benchmark", type="primary"):
                if not selected_solvers:
                    st.error("Please select at least one solver.")
                    return
                
                with st.spinner("Running benchmark..."):
                    # Create solvers
                    solvers = []
                    for solver_type in selected_solvers:
                        try:
                            solver = create_ik_solver("planar_2r", solver_type)
                            solvers.append(solver)
                        except Exception as e:
                            st.error(f"Error creating {solver_type}: {str(e)}")
                    
                    if not solvers:
                        st.error("No solvers could be created.")
                        return
                    
                    # Generate test targets
                    evaluator = IKEvaluator(solvers[0].robot_model)
                    test_targets = evaluator.generate_test_targets(n_tests, test_distribution)
                    
                    # Run benchmark
                    benchmark_results = evaluator.compare_solvers(solvers, test_targets)
                    st.session_state.benchmark_results = benchmark_results
                
                st.success("✅ Benchmark completed!")
        
        with col2:
            st.subheader("Benchmark Results")
            
            if st.session_state.benchmark_results:
                self.display_benchmark_results(st.session_state.benchmark_results)
    
    def render_documentation(self):
        """Render the documentation tab."""
        st.header("Documentation")
        
        st.markdown("""
        ## Inverse Kinematics Solver
        
        This application provides interactive testing and visualization of various 
        inverse kinematics algorithms for robotic manipulators.
        
        ### Features
        
        - **Multiple Robot Models**: 2R planar, 3R planar, and 6R spatial manipulators
        - **Various IK Solvers**: Analytical, Newton-Raphson, Levenberg-Marquardt, 
          Jacobian Transpose, and Redundant IK methods
        - **Interactive Testing**: Real-time IK solving with visualization
        - **Trajectory Planning**: Support for circular, linear, and custom trajectories
        - **Comprehensive Benchmarking**: Performance evaluation with multiple metrics
        
        ### IK Solver Methods
        
        1. **Analytical 2R**: Closed-form solution for 2-DOF planar arms
        2. **Newton-Raphson**: Iterative method using Jacobian pseudo-inverse
        3. **Levenberg-Marquardt**: Robust iterative method with adaptive damping
        4. **Jacobian Transpose**: Simple gradient-based method
        5. **Redundant IK**: Handles redundant manipulators with null-space optimization
        
        ### Usage
        
        1. Select robot type and IK solver in the sidebar
        2. Configure robot and solver parameters
        3. Click "Create Solver" to initialize
        4. Use the Interactive Testing tab to solve individual targets
        5. Use the Trajectory Planning tab for path following
        6. Use the Benchmarking tab to compare solver performance
        
        ### Safety Notice
        
        This software is for educational and research purposes only. 
        Do not use on real hardware without proper safety measures and expert review.
        """)
    
    def plot_robot_configuration(self, target_pose: np.ndarray, joint_angles: np.ndarray = None):
        """Plot robot configuration."""
        fig = go.Figure()
        
        # Plot workspace boundary
        if hasattr(st.session_state.ik_solver.robot_model, 'workspace_radius'):
            min_radius, max_radius = st.session_state.ik_solver.robot_model.workspace_radius()
            
            # Outer boundary
            theta = np.linspace(0, 2*np.pi, 100)
            x_outer = max_radius * np.cos(theta)
            y_outer = max_radius * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_outer, y=y_outer,
                mode='lines',
                name='Workspace Boundary',
                line=dict(color='gray', dash='dash')
            ))
            
            # Inner boundary
            if min_radius > 0:
                x_inner = min_radius * np.cos(theta)
                y_inner = min_radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x_inner, y=y_inner,
                    mode='lines',
                    name='Inner Boundary',
                    line=dict(color='gray', dash='dash')
                ))
        
        # Plot target
        fig.add_trace(go.Scatter(
            x=[target_pose[0]], y=[target_pose[1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Target'
        ))
        
        # Plot robot configuration
        if joint_angles is not None:
            robot_model = st.session_state.ik_solver.robot_model
            current_pose = robot_model.forward_kinematics(joint_angles)
            
            # Plot end-effector
            fig.add_trace(go.Scatter(
                x=[current_pose[0]], y=[current_pose[1]],
                mode='markers',
                marker=dict(size=10, color='blue'),
                name='End-Effector'
            ))
            
            # Plot robot links (simplified for 2R)
            if hasattr(robot_model, 'link_lengths') and len(robot_model.link_lengths) == 2:
                l1, l2 = robot_model.link_lengths
                theta1, theta2 = joint_angles
                
                # Joint positions
                joint1_pos = [0, 0]
                joint2_pos = [l1 * np.cos(theta1), l1 * np.sin(theta1)]
                end_pos = [current_pose[0], current_pose[1]]
                
                # Plot links
                fig.add_trace(go.Scatter(
                    x=[joint1_pos[0], joint2_pos[0]], 
                    y=[joint1_pos[1], joint2_pos[1]],
                    mode='lines+markers',
                    line=dict(width=5, color='blue'),
                    marker=dict(size=8, color='blue'),
                    name='Link 1'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[joint2_pos[0], end_pos[0]], 
                    y=[joint2_pos[1], end_pos[1]],
                    mode='lines+markers',
                    line=dict(width=5, color='green'),
                    marker=dict(size=8, color='green'),
                    name='Link 2'
                ))
        
        fig.update_layout(
            title="Robot Configuration",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            showlegend=True,
            aspectratio=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_trajectory_results(self, trajectory: np.ndarray, results: List):
        """Plot trajectory results."""
        fig = go.Figure()
        
        # Plot trajectory
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0], y=trajectory[:, 1],
            mode='lines+markers',
            name='Target Trajectory',
            line=dict(color='red', width=2),
            marker=dict(size=6, color='red')
        ))
        
        # Plot successful solutions
        successful_results = [r for r in results if r.success]
        if successful_results:
            actual_poses = []
            for result in successful_results:
                actual_pose = st.session_state.ik_solver.robot_model.forward_kinematics(result.joint_angles)
                actual_poses.append(actual_pose)
            
            actual_poses = np.array(actual_poses)
            
            fig.add_trace(go.Scatter(
                x=actual_poses[:, 0], y=actual_poses[:, 1],
                mode='lines+markers',
                name='Actual Trajectory',
                line=dict(color='blue', width=2),
                marker=dict(size=6, color='blue')
            ))
        
        fig.update_layout(
            title="Trajectory Execution",
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            showlegend=True,
            aspectratio=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_benchmark_results(self, benchmark_results: Dict[str, Any]):
        """Display benchmark results."""
        # Summary table
        df_data = []
        for solver_name, result in benchmark_results.items():
            df_data.append({
                'Solver': solver_name,
                'Success Rate': f"{result.success_rate:.3f}",
                'Avg Error': f"{result.average_error:.2e}",
                'Avg Time (ms)': f"{result.average_time*1000:.2f}",
                'Robustness': f"{result.robustness_score:.3f}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Success Rate', 'Average Error', 'Average Time', 'Robustness Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        solver_names = list(benchmark_results.keys())
        
        # Success rate
        fig.add_trace(
            go.Bar(x=solver_names, y=[r.success_rate for r in benchmark_results.values()]),
            row=1, col=1
        )
        
        # Average error
        fig.add_trace(
            go.Bar(x=solver_names, y=[r.average_error for r in benchmark_results.values()]),
            row=1, col=2
        )
        
        # Average time
        fig.add_trace(
            go.Bar(x=solver_names, y=[r.average_time*1000 for r in benchmark_results.values()]),
            row=2, col=1
        )
        
        # Robustness score
        fig.add_trace(
            go.Bar(x=solver_names, y=[r.robustness_score for r in benchmark_results.values()]),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    app = IKVisualizationApp()
    app.run()


if __name__ == "__main__":
    main()
