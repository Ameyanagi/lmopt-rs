//! Comprehensive tests for global optimization methods.
//!
//! These tests evaluate the performance and correctness of the global optimization
//! algorithms on standard test functions with known global minima.

use approx::assert_relative_eq;
use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::{
    optimize_global, optimize_global_parallel, BasinHopping, DifferentialEvolution,
    GlobalOptimizer, HybridGlobal, ParallelDifferentialEvolution, SimulatedAnnealing,
};
use lmopt_rs::problem::Problem;
use ndarray::{array, Array1};
use std::f64::consts::PI;

// --- Standard Test Functions for Global Optimization ---

/// Rastrigin function: f(x) = 10n + sum[x_i^2 - 10cos(2πx_i)]
/// Global minimum at x_i = 0 for all i, with f(x) = 0
struct RastriginFunction {
    dimension: usize,
}

impl RastriginFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for RastriginFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let mut sum = 10.0 * self.dimension as f64;
        for x in params.iter() {
            sum += x.powi(2) - 10.0 * (2.0 * PI * x).cos();
        }

        Ok(array![sum.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Rosenbrock function: f(x) = sum[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
/// Global minimum at x_i = 1 for all i, with f(x) = 0
struct RosenbrockFunction {
    dimension: usize,
}

impl RosenbrockFunction {
    fn new(dimension: usize) -> Self {
        if dimension < 2 {
            panic!("Rosenbrock function requires at least 2 dimensions");
        }
        Self { dimension }
    }
}

impl Problem for RosenbrockFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let mut sum = 0.0;
        for i in 0..self.dimension - 1 {
            let term1 = 100.0 * (params[i + 1] - params[i].powi(2)).powi(2);
            let term2 = (1.0 - params[i]).powi(2);
            sum += term1 + term2;
        }

        Ok(array![sum.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Ackley function: f(x) = -20exp(-0.2√(1/n∑x_i^2)) - exp(1/n∑cos(2πx_i)) + 20 + e
/// Global minimum at x_i = 0 for all i, with f(x) = 0
struct AckleyFunction {
    dimension: usize,
}

impl AckleyFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for AckleyFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let n = self.dimension as f64;

        // First term
        let sum1: f64 = params.iter().map(|&x| x.powi(2)).sum::<f64>() / n;
        let term1 = -20.0 * (-0.2 * sum1.sqrt()).exp();

        // Second term
        let sum2: f64 = params.iter().map(|&x| (2.0 * PI * x).cos()).sum::<f64>() / n;
        let term2 = -sum2.exp();

        // Complete function
        let result = term1 + term2 + 20.0 + std::f64::consts::E;

        Ok(array![result])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Levy function: f(x) = sin^2(πw_1) + sum[(w_i-1)^2(1+10sin^2(πw_i+1))] + (w_d-1)^2(1+sin^2(2πw_d))
/// where w_i = 1 + (x_i - 1)/4
/// Global minimum at x_i = 1 for all i, with f(x) = 0
struct LevyFunction {
    dimension: usize,
}

impl LevyFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for LevyFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        // Transform parameters to w
        let w: Vec<f64> = params.iter().map(|&x| 1.0 + (x - 1.0) / 4.0).collect();

        // First term
        let term1 = (PI * w[0]).sin().powi(2);

        // Middle terms
        let mut sum = 0.0;
        for i in 0..(self.dimension - 1) {
            sum += (w[i] - 1.0).powi(2) * (1.0 + 10.0 * (PI * w[i] + 1.0).sin().powi(2));
        }

        // Last term
        let last_idx = self.dimension - 1;
        let term3 = (w[last_idx] - 1.0).powi(2) * (1.0 + (2.0 * PI * w[last_idx]).sin().powi(2));

        let result = term1 + sum + term3;

        Ok(array![result.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

// --- Helper Functions ---

/// Run the optimizer and check if it finds a solution close to the expected minimum
fn check_optimization<P: Problem, O: GlobalOptimizer>(
    problem: &P,
    optimizer: &O,
    bounds: &[(f64, f64)],
    expected_min: &[f64],
    tolerance: f64,
) -> Result<()> {
    let result = optimizer.optimize(problem, bounds, 100, 30, 1e-6)?;

    println!("Optimizer: {}", std::any::type_name::<O>());
    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Solution: {:?}", result.params);
    println!("  Expected: {:?}", expected_min);

    // Check if the solution is close to the expected minimum
    assert!(result.success, "Optimization did not succeed");

    // For each parameter, check if it's close to the expected value
    for (i, (&found, &expected)) in result.params.iter().zip(expected_min.iter()).enumerate() {
        assert_relative_eq!(
            found,
            expected,
            epsilon = tolerance,
            max_relative = tolerance.powi(2),
            "Parameter {} not close to expected: {} != {}",
            i,
            found,
            expected
        );
    }

    // Check if the cost is close to zero
    assert!(
        result.cost < tolerance,
        "Cost not close to minimum: {}",
        result.cost
    );

    Ok(())
}

// --- Tests for Different Optimizers ---

#[test]
fn test_simulated_annealing_rastrigin() -> Result<()> {
    let dim = 2;
    let problem = RastriginFunction::new(dim);
    let bounds = vec![(-5.12, 5.12); dim];
    let expected_min = vec![0.0; dim];

    let optimizer = SimulatedAnnealing::new()
        .with_initial_temperature(10.0)
        .with_cooling_rate(0.95);

    check_optimization(&problem, &optimizer, &bounds, &expected_min, 0.5)
}

#[test]
fn test_differential_evolution_rastrigin() -> Result<()> {
    let dim = 5;
    let problem = RastriginFunction::new(dim);
    let bounds = vec![(-5.12, 5.12); dim];
    let expected_min = vec![0.0; dim];

    let optimizer = DifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8);

    check_optimization(&problem, &optimizer, &bounds, &expected_min, 0.1)
}

#[test]
fn test_parallel_differential_evolution_rastrigin() -> Result<()> {
    let dim = 5;
    let problem = RastriginFunction::new(dim);
    let bounds = vec![(-5.12, 5.12); dim];
    let expected_min = vec![0.0; dim];

    let optimizer = ParallelDifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8);

    check_optimization(&problem, &optimizer, &bounds, &expected_min, 0.1)
}

#[test]
fn test_basin_hopping_rosenbrock() -> Result<()> {
    let dim = 2;
    let problem = RosenbrockFunction::new(dim);
    let bounds = vec![(-5.0, 10.0); dim];
    let expected_min = vec![1.0; dim];

    let optimizer = BasinHopping::new()
        .with_step_size(0.5)
        .with_local_optimization(true);

    check_optimization(&problem, &optimizer, &bounds, &expected_min, 0.1)
}

#[test]
fn test_hybrid_global_ackley() -> Result<()> {
    let dim = 3;
    let problem = AckleyFunction::new(dim);
    let bounds = vec![(-5.0, 5.0); dim];
    let expected_min = vec![0.0; dim];

    let optimizer = HybridGlobal::new();

    check_optimization(&problem, &optimizer, &bounds, &expected_min, 0.1)
}

#[test]
fn test_optimize_global_levy() -> Result<()> {
    let dim = 4;
    let problem = LevyFunction::new(dim);
    let bounds = vec![(-10.0, 10.0); dim];
    let expected_min = vec![1.0; dim];

    // Use the convenience function
    let result = optimize_global(&problem, &bounds, 100, 40, 1e-6)?;

    println!("optimize_global result:");
    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!("  Iterations: {}", result.iterations);
    println!("  Solution: {:?}", result.params);

    assert!(result.success, "Optimization did not succeed");

    // Check if the solution is close to the expected minimum
    for (i, (&found, &expected)) in result.params.iter().zip(expected_min.iter()).enumerate() {
        assert_relative_eq!(
            found,
            expected,
            epsilon = 0.5,
            "Parameter {} not close to expected: {} != {}",
            i,
            found,
            expected
        );
    }

    Ok(())
}

#[test]
fn test_optimize_global_parallel_levy() -> Result<()> {
    let dim = 4;
    let problem = LevyFunction::new(dim);
    let bounds = vec![(-10.0, 10.0); dim];
    let expected_min = vec![1.0; dim];

    // Use the parallel convenience function
    let result = optimize_global_parallel(&problem, &bounds, 100, 40, 1e-6)?;

    println!("optimize_global_parallel result:");
    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!("  Iterations: {}", result.iterations);
    println!("  Solution: {:?}", result.params);

    assert!(result.success, "Optimization did not succeed");

    // Check if the solution is close to the expected minimum
    for (i, (&found, &expected)) in result.params.iter().zip(expected_min.iter()).enumerate() {
        assert_relative_eq!(
            found,
            expected,
            epsilon = 0.5,
            "Parameter {} not close to expected: {} != {}",
            i,
            found,
            expected
        );
    }

    Ok(())
}

// --- Parameter Problem Tests ---

#[test]
fn test_multiple_dimensions() {
    // Test each optimizer on problems with increasing dimensionality
    let dimensions = [2, 5, 10];
    let tolerance = 0.5; // Higher tolerance for higher dimensions

    for &dim in &dimensions {
        println!("\nTesting dimension: {}", dim);

        // Rastrigin function (challenging for most optimizers)
        let problem = RastriginFunction::new(dim);
        let bounds = vec![(-5.12, 5.12); dim];
        let expected_min = vec![0.0; dim];

        // Different optimizers
        let de = DifferentialEvolution::new()
            .with_population_size(40 + dim * 5)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8);

        let res = de.optimize(&problem, &bounds, 150, 50, 1e-6);
        match res {
            Ok(result) => {
                println!(
                    "DifferentialEvolution - dim {}: cost = {:.6e}",
                    dim, result.cost
                );
                // Be more lenient with higher dimensions
                let max_distance = result
                    .params
                    .iter()
                    .zip(expected_min.iter())
                    .map(|(&p, &e)| (p - e).abs())
                    .fold(0.0, f64::max);

                assert!(
                    max_distance < tolerance * (dim as f64).sqrt(),
                    "Solution too far from minimum: {:.6}",
                    max_distance
                );
            }
            Err(e) => {
                panic!("Optimization failed: {}", e);
            }
        }

        // Test parallel DE with higher dimensions (more likely to benefit from parallelism)
        if dim >= 5 {
            let par_de = ParallelDifferentialEvolution::new()
                .with_population_size(40 + dim * 5)
                .with_crossover_probability(0.9)
                .with_differential_weight(0.8);

            let res = par_de.optimize(&problem, &bounds, 150, 50, 1e-6);
            match res {
                Ok(result) => {
                    println!(
                        "ParallelDifferentialEvolution - dim {}: cost = {:.6e}",
                        dim, result.cost
                    );

                    let max_distance = result
                        .params
                        .iter()
                        .zip(expected_min.iter())
                        .map(|(&p, &e)| (p - e).abs())
                        .fold(0.0, f64::max);

                    assert!(
                        max_distance < tolerance * (dim as f64).sqrt(),
                        "Solution too far from minimum: {:.6}",
                        max_distance
                    );
                }
                Err(e) => {
                    panic!("Parallel optimization failed: {}", e);
                }
            }
        }
    }
}

// --- Boundary Handling Tests ---

#[test]
fn test_boundary_handling() -> Result<()> {
    // Test if optimizers respect parameter bounds
    let dim = 3;
    let problem = RastriginFunction::new(dim);

    // Asymmetric bounds to make the test more challenging
    let bounds = vec![
        (-1.0, 3.0), // Parameter 1: between -1 and 3
        (-5.0, 0.0), // Parameter 2: between -5 and 0
        (0.0, 2.0),  // Parameter 3: between 0 and 2
    ];

    // Global minimum at [0, 0, 0] - within bounds

    // Test different optimizers
    let optimizers: Vec<Box<dyn GlobalOptimizer>> = vec![
        Box::new(DifferentialEvolution::new().with_population_size(30)),
        Box::new(ParallelDifferentialEvolution::new().with_population_size(30)),
        Box::new(BasinHopping::new()),
        Box::new(SimulatedAnnealing::new()),
        Box::new(HybridGlobal::new()),
    ];

    for optimizer in optimizers {
        let result = optimizer.optimize(&problem, &bounds, 100, 30, 1e-5)?;

        println!(
            "Checking bounds for: {}",
            std::any::type_name::<dyn GlobalOptimizer>()
        );

        // Check if all parameters are within bounds
        for (i, (&param, &(min, max))) in result.params.iter().zip(bounds.iter()).enumerate() {
            assert!(
                param >= min && param <= max,
                "Parameter {} out of bounds: {} not in [{}, {}]",
                i,
                param,
                min,
                max
            );
        }
    }

    Ok(())
}

// --- Robustness Tests ---

#[test]
fn test_multiple_starting_points() -> Result<()> {
    // Test if the optimizers are robust to different initial conditions
    let problem = RosenbrockFunction::new(2); // Challenging function
    let bounds = vec![(-5.0, 10.0); 2];
    let expected_min = vec![1.0; 2];

    // Run multiple optimizations with different seeds
    let seeds = [42, 123, 987, 555, 7890];

    for &seed in &seeds {
        let de = DifferentialEvolution::new()
            .with_population_size(30)
            .with_seed(seed);

        let result = de.optimize(&problem, &bounds, 100, 30, 1e-6)?;

        println!(
            "Seed {}: cost = {:.6e}, params = {:?}",
            seed, result.cost, result.params
        );

        // Verify convergence to the global minimum
        assert!(
            result.success,
            "Optimization did not succeed with seed {}",
            seed
        );

        // Check if the solution is close to [1, 1]
        for (i, (&found, &expected)) in result.params.iter().zip(expected_min.iter()).enumerate() {
            assert_relative_eq!(
                found,
                expected,
                epsilon = 0.1,
                "Parameter {} not close to expected with seed {}: {} != {}",
                i,
                seed,
                found,
                expected
            );
        }
    }

    Ok(())
}

// --- Optimization with Local Refinement Tests ---

#[test]
fn test_local_optimization() -> Result<()> {
    // Test if local optimization step improves results
    let dim = 4;
    let problem = LevyFunction::new(dim);
    let bounds = vec![(-10.0, 10.0); dim];

    // Run without local optimization
    let de_without_local = DifferentialEvolution::new()
        .with_population_size(40)
        .with_local_optimization(false);

    let result_without_local = de_without_local.optimize(&problem, &bounds, 80, 30, 1e-6)?;

    // Run with local optimization
    let de_with_local = DifferentialEvolution::new()
        .with_population_size(40)
        .with_local_optimization(true)
        .with_local_optimization_iterations(100);

    let result_with_local = de_with_local.optimize(&problem, &bounds, 80, 30, 1e-6)?;

    println!(
        "Without local optimization: cost = {:.6e}",
        result_without_local.cost
    );
    println!(
        "With local optimization: cost = {:.6e}",
        result_with_local.cost
    );

    // Local optimization should generally improve results
    assert!(
        result_with_local.cost <= result_without_local.cost * 1.05,
        "Local optimization did not improve results significantly"
    );

    Ok(())
}

// --- Comparing Different Algorithms ---

#[test]
fn compare_all_algorithms() -> Result<()> {
    // Compare all global optimization algorithms on a standard test function
    let dim = 3;
    let problem = RastriginFunction::new(dim);
    let bounds = vec![(-5.12, 5.12); dim];

    // List of all optimizers
    let optimizers: Vec<Box<dyn GlobalOptimizer>> = vec![
        Box::new(SimulatedAnnealing::new().with_initial_temperature(10.0)),
        Box::new(BasinHopping::new().with_step_size(0.5)),
        Box::new(DifferentialEvolution::new().with_population_size(40)),
        Box::new(ParallelDifferentialEvolution::new().with_population_size(40)),
        Box::new(HybridGlobal::new()),
    ];

    // Track results
    let mut results = Vec::new();

    // Run all optimizers
    for optimizer in optimizers {
        let type_name = std::any::type_name::<dyn GlobalOptimizer>();
        let name = type_name.split("::").last().unwrap_or(type_name);

        let result = optimizer.optimize(&problem, &bounds, 100, 30, 1e-6)?;

        println!("Optimizer: {}", name);
        println!("  Success: {}", result.success);
        println!("  Cost: {:.6e}", result.cost);
        println!("  Iterations: {}", result.iterations);
        println!("  Function evaluations: {}", result.func_evals);

        results.push((
            name.to_string(),
            result.success,
            result.cost,
            result.func_evals,
        ));
    }

    // Find best optimizer by cost
    let best = results
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap();

    println!("\nBest optimizer: {} with cost {:.6e}", best.0, best.2);

    // Ensure at least one optimizer succeeded
    assert!(results.iter().any(|r| r.1), "No optimizer succeeded");

    Ok(())
}
