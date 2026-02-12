"""
Evaluation metrics and benchmarking for inverse kinematics.

This module provides comprehensive evaluation metrics for IK solvers including
accuracy, efficiency, robustness, and workspace coverage metrics.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
from dataclasses import dataclass
from .kinematics import UnifiedIKSolver, IKResult


@dataclass
class IKBenchmarkResult:
    """Container for IK benchmark results."""
    
    solver_name: str
    success_rate: float
    average_error: float
    average_time: float
    average_iterations: float
    workspace_coverage: float
    robustness_score: float
    total_tests: int
    successful_tests: int
    failed_tests: int
    error_distribution: Dict[str, float]
    timing_distribution: Dict[str, float]


class IKEvaluator:
    """
    Comprehensive evaluator for inverse kinematics solvers.
    
    Provides various metrics for evaluating IK solver performance including
    accuracy, efficiency, robustness, and workspace coverage.
    """
    
    def __init__(self, robot_model, workspace_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Initialize IK evaluator.
        
        Args:
            robot_model: Robot kinematic model
            workspace_bounds: Optional workspace bounds (min_bounds, max_bounds)
        """
        self.robot_model = robot_model
        self.workspace_bounds = workspace_bounds
    
    def generate_test_targets(self, n_targets: int = 1000, 
                            distribution: str = 'uniform') -> List[np.ndarray]:
        """
        Generate test target poses for evaluation.
        
        Args:
            n_targets: Number of target poses to generate
            distribution: Distribution type ('uniform', 'gaussian', 'edge_case')
            
        Returns:
            List of target poses
        """
        targets = []
        
        if distribution == 'uniform':
            if self.workspace_bounds is not None:
                min_bounds, max_bounds = self.workspace_bounds
                for _ in range(n_targets):
                    target = np.random.uniform(min_bounds, max_bounds)
                    targets.append(target)
            else:
                # Default uniform distribution for 2D workspace
                for _ in range(n_targets):
                    # Generate random points in unit circle
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(0.1, 1.9)  # Within workspace
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    targets.append(np.array([x, y]))
        
        elif distribution == 'gaussian':
            # Gaussian distribution around workspace center
            center = np.zeros(2) if self.workspace_bounds is None else np.mean(self.workspace_bounds, axis=0)
            std = 0.5
            for _ in range(n_targets):
                target = np.random.normal(center, std)
                targets.append(target)
        
        elif distribution == 'edge_case':
            # Edge cases: workspace boundaries, singularities
            if hasattr(self.robot_model, 'workspace_radius'):
                min_radius, max_radius = self.robot_model.workspace_radius()
                
                # Boundary cases
                for angle in np.linspace(0, 2*np.pi, n_targets//3):
                    x = max_radius * np.cos(angle)
                    y = max_radius * np.sin(angle)
                    targets.append(np.array([x, y]))
                
                # Near boundary
                for angle in np.linspace(0, 2*np.pi, n_targets//3):
                    radius = max_radius * 0.95
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    targets.append(np.array([x, y]))
                
                # Center region
                for _ in range(n_targets - 2*(n_targets//3)):
                    angle = np.random.uniform(0, 2*np.pi)
                    radius = np.random.uniform(min_radius, min_radius + 0.1)
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    targets.append(np.array([x, y]))
        
        return targets
    
    def evaluate_solver(self, solver: UnifiedIKSolver, test_targets: List[np.ndarray],
                       initial_guesses: Optional[List[np.ndarray]] = None) -> IKBenchmarkResult:
        """
        Evaluate an IK solver comprehensively.
        
        Args:
            solver: IK solver to evaluate
            test_targets: List of target poses to test
            initial_guesses: Optional list of initial guesses
            
        Returns:
            Comprehensive benchmark result
        """
        results = []
        computation_times = []
        errors = []
        
        # Run tests
        for i, target in enumerate(test_targets):
            initial_guess = initial_guesses[i] if initial_guesses else None
            
            start_time = time.time()
            result = solver.solve(target, initial_guess)
            computation_times.append(time.time() - start_time)
            
            results.append(result)
            errors.append(result.error if result.success else float('inf'))
        
        # Calculate metrics
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results) if results else 0
        
        if successful_results:
            avg_error = np.mean([r.error for r in successful_results])
            avg_time = np.mean([r.computation_time for r in successful_results])
            avg_iterations = np.mean([r.iterations for r in successful_results])
            
            # Error distribution
            error_percentiles = np.percentile([r.error for r in successful_results], [25, 50, 75, 90, 95, 99])
            error_distribution = {
                'p25': error_percentiles[0],
                'p50': error_percentiles[1],
                'p75': error_percentiles[2],
                'p90': error_percentiles[3],
                'p95': error_percentiles[4],
                'p99': error_percentiles[5]
            }
            
            # Timing distribution
            time_percentiles = np.percentile([r.computation_time for r in successful_results], [25, 50, 75, 90, 95, 99])
            timing_distribution = {
                'p25': time_percentiles[0],
                'p50': time_percentiles[1],
                'p75': time_percentiles[2],
                'p90': time_percentiles[3],
                'p95': time_percentiles[4],
                'p99': time_percentiles[5]
            }
        else:
            avg_error = float('inf')
            avg_time = float('inf')
            avg_iterations = float('inf')
            error_distribution = {}
            timing_distribution = {}
        
        # Workspace coverage
        workspace_coverage = self._calculate_workspace_coverage(results, test_targets)
        
        # Robustness score
        robustness_score = self._calculate_robustness_score(results)
        
        return IKBenchmarkResult(
            solver_name=solver.solver.method,
            success_rate=success_rate,
            average_error=avg_error,
            average_time=avg_time,
            average_iterations=avg_iterations,
            workspace_coverage=workspace_coverage,
            robustness_score=robustness_score,
            total_tests=len(results),
            successful_tests=len(successful_results),
            failed_tests=len(results) - len(successful_results),
            error_distribution=error_distribution,
            timing_distribution=timing_distribution
        )
    
    def _calculate_workspace_coverage(self, results: List[IKResult], 
                                    test_targets: List[np.ndarray]) -> float:
        """
        Calculate workspace coverage percentage.
        
        Args:
            results: List of IK results
            test_targets: List of test targets
            
        Returns:
            Workspace coverage percentage
        """
        if not results:
            return 0.0
        
        successful_targets = [test_targets[i] for i, result in enumerate(results) if result.success]
        
        if not successful_targets:
            return 0.0
        
        # For 2D workspace, calculate coverage as percentage of reachable area
        if len(test_targets[0]) == 2:
            # Calculate convex hull of successful targets
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(successful_targets)
                covered_area = hull.volume  # For 2D, volume is area
                
                # Estimate total workspace area
                if hasattr(self.robot_model, 'workspace_radius'):
                    min_radius, max_radius = self.robot_model.workspace_radius()
                    total_area = np.pi * (max_radius**2 - min_radius**2)
                    return min(covered_area / total_area, 1.0)
                else:
                    # Estimate from test targets
                    all_points = np.array(test_targets)
                    all_hull = ConvexHull(all_points)
                    total_area = all_hull.volume
                    return min(covered_area / total_area, 1.0)
            except:
                # Fallback: simple ratio
                return len(successful_targets) / len(test_targets)
        
        return len(successful_targets) / len(test_targets)
    
    def _calculate_robustness_score(self, results: List[IKResult]) -> float:
        """
        Calculate robustness score based on error consistency.
        
        Args:
            results: List of IK results
            
        Returns:
            Robustness score (0-1)
        """
        successful_results = [r for r in results if r.success]
        
        if len(successful_results) < 2:
            return 0.0
        
        errors = [r.error for r in successful_results]
        
        # Robustness based on error consistency (lower variance is better)
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        
        if error_mean == 0:
            return 1.0
        
        # Coefficient of variation (lower is better)
        cv = error_std / error_mean
        
        # Convert to 0-1 score (lower CV = higher robustness)
        robustness = max(0, 1 - cv)
        
        return min(robustness, 1.0)
    
    def compare_solvers(self, solvers: List[UnifiedIKSolver], 
                       test_targets: List[np.ndarray]) -> Dict[str, IKBenchmarkResult]:
        """
        Compare multiple IK solvers.
        
        Args:
            solvers: List of IK solvers to compare
            test_targets: List of target poses to test
            
        Returns:
            Dictionary of benchmark results for each solver
        """
        results = {}
        
        for solver in solvers:
            result = self.evaluate_solver(solver, test_targets)
            results[solver.solver.method] = result
        
        return results
    
    def generate_performance_report(self, benchmark_results: Dict[str, IKBenchmarkResult]) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            benchmark_results: Dictionary of benchmark results
            
        Returns:
            Formatted performance report
        """
        report = "Inverse Kinematics Solver Performance Report\n"
        report += "=" * 50 + "\n\n"
        
        # Summary table
        report += "Summary:\n"
        report += f"{'Solver':<20} {'Success Rate':<12} {'Avg Error':<12} {'Avg Time (ms)':<15} {'Robustness':<12}\n"
        report += "-" * 80 + "\n"
        
        for solver_name, result in benchmark_results.items():
            report += f"{solver_name:<20} {result.success_rate:<12.3f} {result.average_error:<12.2e} "
            report += f"{result.average_time*1000:<15.2f} {result.robustness_score:<12.3f}\n"
        
        report += "\n"
        
        # Detailed results for each solver
        for solver_name, result in benchmark_results.items():
            report += f"{solver_name} Detailed Results:\n"
            report += f"  Success Rate: {result.success_rate:.3f} ({result.successful_tests}/{result.total_tests})\n"
            report += f"  Average Error: {result.average_error:.2e}\n"
            report += f"  Average Time: {result.average_time*1000:.2f} ms\n"
            report += f"  Average Iterations: {result.average_iterations:.1f}\n"
            report += f"  Workspace Coverage: {result.workspace_coverage:.3f}\n"
            report += f"  Robustness Score: {result.robustness_score:.3f}\n"
            
            if result.error_distribution:
                report += f"  Error Distribution:\n"
                report += f"    P50: {result.error_distribution['p50']:.2e}\n"
                report += f"    P95: {result.error_distribution['p95']:.2e}\n"
                report += f"    P99: {result.error_distribution['p99']:.2e}\n"
            
            if result.timing_distribution:
                report += f"  Timing Distribution:\n"
                report += f"    P50: {result.timing_distribution['p50']*1000:.2f} ms\n"
                report += f"    P95: {result.timing_distribution['p95']*1000:.2f} ms\n"
                report += f"    P99: {result.timing_distribution['p99']*1000:.2f} ms\n"
            
            report += "\n"
        
        return report
