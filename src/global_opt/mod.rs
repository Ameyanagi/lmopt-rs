//! # Global Optimization
//!
//! This module provides methods for global optimization to find the global minimum
//! of nonlinear problems with multiple local minima. It offers several powerful
//! algorithms and a unified interface for solving complex optimization challenges.
//!
//! ## Key Features
//!
//! - **Multiple Algorithms**: Several complementary approaches for different problems
//! - **Parameter Bounds**: Support for box constraints on parameters
//! - **Unified Interface**: Common API for all optimization methods
//! - **Integration with Parameters**: Works with the parameter system for named parameters
//! - **Hybrid Methods**: Combines global exploration with local refinement
//!
//! ## Optimization Algorithms
//!
//! - **[`SimulatedAnnealing`]**: Mimics the annealing process in metallurgy
//! - **[`DifferentialEvolution`]**: Population-based evolutionary algorithm
//! - **[`BasinHopping`]**: Combines random jumps with local optimization
//! - **[`HybridGlobal`]**: Meta-optimizer that combines multiple approaches
//!
//! ## Example Usage
//!
//! ```rust
//! use lmopt_rs::{optimize_global, Problem};
//! use ndarray::{array, Array1};
//!
//! // Define a problem with multiple local minima
//! struct MultiWellProblem;
//!
//! impl Problem for MultiWellProblem {
//!     fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
//!         let x = params[0];
//!         let y = params[1];
//!         
//!         // Rastrigin function: has multiple local minima
//!         // f(x, y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))
//!         let term1 = 20.0 + x.powi(2) + y.powi(2);
//!         let term2 = 10.0 * ((2.0 * std::f64::consts::PI * x).cos() +
//!                            (2.0 * std::f64::consts::PI * y).cos());
//!         let value = term1 - term2;
//!         
//!         Ok(array![value.sqrt()])
//!     }
//!     
//!     fn parameter_count(&self) -> usize { 2 }
//!     
//!     fn residual_count(&self) -> usize { 1 }
//! }
//!
//! // Create the problem
//! let problem = MultiWellProblem;
//!
//! // Define bounds for the parameters
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//!
//! // Run global optimization
//! let result = optimize_global(
//!     &problem,    // Problem to solve
//!     &bounds,     // Parameter bounds
//!     100,         // Number of iterations
//!     20,          // Population size
//!     1e-6         // Tolerance
//! ).unwrap();
//!
//! println!("Global minimum found at: ({:.3}, {:.3})",
//!          result.params[0], result.params[1]);
//! println!("Function value: {:.6}", result.cost);
//! ```
//!
//! ## Using Specific Algorithms
//!
//! ```rust
//! use lmopt_rs::{GlobalOptimizer, DifferentialEvolution, BasinHopping};
//!
//! // Differential Evolution
//! let de = DifferentialEvolution::new()
//!     .with_crossover_probability(0.7)
//!     .with_differential_weight(0.8);
//! let de_result = de.optimize(&problem, &bounds, 100, 30, 1e-6).unwrap();
//!
//! // Basin Hopping
//! let bh = BasinHopping::new()
//!     .with_step_size(1.0)
//!     .with_temperature(1.0);
//! let bh_result = bh.optimize(&problem, &bounds, 50, 10, 1e-6).unwrap();
//! ```
//!
//! For comprehensive documentation, see the [Global Optimization guide](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/global_optimization.md).

use ndarray::{Array1, Array2};
use rand::Rng;
use std::fmt;

use crate::error::{LmOptError, Result};
use crate::lm::{LevenbergMarquardt, LmConfig, LmResult};
use crate::problem::Problem;
use crate::problem_params::ParameterProblem;

/// Trait for global optimization methods.
///
/// This trait defines the interface for all global optimization algorithms.
/// Implementing this trait allows an optimizer to work with the common
/// optimization interface and be used with the utility functions in this module.
///
/// The main method to implement is `optimize`, which performs the optimization
/// on a given problem with specified bounds and convergence criteria.
pub trait GlobalOptimizer {
    /// Run the optimization and return the best solution found.
    ///
    /// This method performs global optimization on the provided problem
    /// within the specified parameter bounds, searching for the global minimum.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem to solve, implementing the `Problem` trait
    /// * `bounds` - Lower and upper bounds for each parameter as Vec of (min, max) tuples
    /// * `max_iterations` - Maximum number of iterations to perform
    /// * `max_no_improvement` - Maximum number of iterations without improvement before stopping
    /// * `tol` - Tolerance for convergence (algorithm-specific usage)
    ///
    /// # Returns
    ///
    /// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use lmopt_rs::{GlobalOptimizer, SimulatedAnnealing};
    /// use lmopt_rs::problem::Problem;
    /// use ndarray::Array1;
    ///
    /// // Your problem implementation
    /// struct MyProblem { /* ... */ }
    /// impl Problem for MyProblem { /* ... */ }
    ///
    /// let problem = MyProblem { /* ... */ };
    /// let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
    /// let mut optimizer = SimulatedAnnealing::new();
    ///
    /// // Run the optimization
    /// let result = optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6).unwrap();
    /// println!("Best parameters: {:?}", result.params);
    /// ```
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult>;

    /// Run the optimization for the current problem.
    ///
    /// This method is used by ParameterProblem implementations that also implement
    /// GlobalOptimizer, allowing them to optimize themselves.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of iterations
    /// * `max_no_improvement` - Maximum number of iterations without improvement
    /// * `tol` - Tolerance for convergence
    ///
    /// # Returns
    ///
    /// * The best solution found and its cost
    fn optimize_param_problem(
        &mut self,
        _max_iterations: usize,
        _max_no_improvement: usize,
        _tol: f64,
    ) -> Result<GlobalOptResult> {
        // Default implementation - can be overridden by implementers
        Err(LmOptError::NotImplemented(
            "optimize_param_problem not implemented for this optimizer".to_string(),
        ))
    }

    /// Run the optimization for an external parameter problem.
    ///
    /// # Arguments
    ///
    /// * `problem` - The parameter problem to solve
    /// * `max_iterations` - Maximum number of iterations
    /// * `max_no_improvement` - Maximum number of iterations without improvement
    /// * `tol` - Tolerance for convergence
    ///
    /// # Returns
    ///
    /// * The best solution found
    fn run_optimization<P: ParameterProblem>(
        &self,
        problem: &mut P,
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        // Get the problem bounds from the parameters
        let bounds = Self::get_bounds_from_parameters(problem)?;

        // Create a parameter problem adapter
        let adapter = crate::problem_params::problem_from_parameter_problem(problem);

        // Run the optimization
        let result = self.optimize(&adapter, &bounds, max_iterations, max_no_improvement, tol)?;

        // Update the problem parameters with the best solution
        problem.update_parameters_from_array(&result.params)?;

        Ok(result)
    }

    /// Get parameter bounds from a parameter problem.
    ///
    /// # Arguments
    ///
    /// * `problem` - The parameter problem
    ///
    /// # Returns
    ///
    /// * A vector of (min, max) bounds for each parameter
    fn get_bounds_from_parameters<P: ParameterProblem>(problem: &P) -> Result<Vec<(f64, f64)>> {
        let mut bounds = Vec::new();

        // Get the parameters
        let params = problem.parameters();

        // Get the varying parameters
        let varying_params = params.varying();

        // Extract bounds for each varying parameter
        for param in varying_params {
            let min = param.min();
            let max = param.max();

            // Handle bounds - if min is -inf or max is inf, use those values
            let effective_min = if min == f64::NEG_INFINITY {
                f64::NEG_INFINITY
            } else {
                min
            };
            let effective_max = if max == f64::INFINITY {
                f64::INFINITY
            } else {
                max
            };

            bounds.push((effective_min, effective_max));
        }

        Ok(bounds)
    }
}

/// Result of a global optimization.
///
/// This struct contains the results of a global optimization run, including
/// the optimal parameters found, the final cost value, and various statistics
/// about the optimization process. It also includes a success flag and
/// descriptive message about the optimization outcome.
///
/// For hybrid methods that combine global and local optimization, the
/// `local_result` field contains the result of the final local optimization.
///
/// # Example
///
/// ```rust,no_run
/// use lmopt_rs::{GlobalOptimizer, SimulatedAnnealing};
/// use lmopt_rs::problem::Problem;
///
/// // Run optimization
/// let mut optimizer = SimulatedAnnealing::new();
/// let result = optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6).unwrap();
///
/// // Access the results
/// if result.success {
///     println!("Optimization succeeded!");
///     println!("Optimal parameters: {:?}", result.params);
///     println!("Final cost: {}", result.cost);
///     println!("Took {} iterations and {} function evaluations",
///              result.iterations, result.func_evals);
/// } else {
///     println!("Optimization failed: {}", result.message);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GlobalOptResult {
    /// The best parameters found by the optimization algorithm
    pub params: Array1<f64>,

    /// The cost (objective function value) at the best parameters
    pub cost: f64,

    /// The number of iterations performed by the optimizer
    pub iterations: usize,

    /// The number of function evaluations performed during optimization
    pub func_evals: usize,

    /// Whether the optimization succeeded in finding a solution
    ///
    /// This is typically true if the algorithm converged within the
    /// specified iteration limits and tolerances.
    pub success: bool,

    /// A descriptive message about the optimization result
    ///
    /// Contains information about why the optimization terminated,
    /// such as convergence, iteration limit, or error conditions.
    pub message: String,

    /// Result of the final local optimization step, if a hybrid method was used
    ///
    /// For methods like Basin Hopping or Hybrid Global-Local optimization,
    /// this contains the result of the final local optimization run.
    pub local_result: Option<LmResult>,
}

impl fmt::Display for GlobalOptResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Global Optimization Result:")?;
        writeln!(f, "  Success: {}", self.success)?;
        writeln!(f, "  Cost: {:.6e}", self.cost)?;
        writeln!(f, "  Iterations: {}", self.iterations)?;
        writeln!(f, "  Function evaluations: {}", self.func_evals)?;
        writeln!(f, "  Message: {}", self.message)?;
        writeln!(f, "  Parameters: {:?}", self.params)?;

        if let Some(ref local_result) = self.local_result {
            writeln!(f, "\nLocal Optimization Result:")?;
            writeln!(f, "  Success: {}", local_result.success)?;
            writeln!(f, "  Cost: {:.6e}", local_result.cost)?;
            writeln!(f, "  Iterations: {}", local_result.iterations)?;
            writeln!(f, "  Message: {}", local_result.message)?;
        }

        Ok(())
    }
}

// Import specific optimization algorithms
mod basin_hopping;
mod differential_evolution;
mod hybrid;
mod parallel;
mod parallel_differential_evolution;
mod simulated_annealing;

// Re-export optimization algorithms
pub use basin_hopping::BasinHopping;
pub use differential_evolution::DifferentialEvolution;
pub use hybrid::HybridGlobal;
pub use parallel::{
    clip_population_parallel, create_population_parallel, evaluate_population_parallel,
};
pub use parallel_differential_evolution::ParallelDifferentialEvolution;
pub use simulated_annealing::SimulatedAnnealing;

/// Run global optimization using a hybrid approach combining global and local methods.
///
/// This is a convenience function that uses the `HybridGlobal` optimizer, which
/// combines global exploration with local refinement using Levenberg-Marquardt.
/// It's an effective general-purpose optimizer for complex problems with
/// multiple local minima.
///
/// # Arguments
///
/// * `problem` - The problem to solve, implementing the `Problem` trait
/// * `bounds` - Lower and upper bounds for each parameter as Vec of (min, max) tuples
/// * `iterations` - Maximum number of iterations for global optimization
/// * `population_size` - Population size (for population-based methods)
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
pub fn optimize_global<P: Problem>(
    problem: &P,
    bounds: &[(f64, f64)],
    iterations: usize,
    population_size: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Create a hybrid optimizer
    let optimizer = HybridGlobal::new();

    // Run the optimization
    optimizer.optimize(problem, bounds, iterations, population_size, tol)
}

/// Run global optimization in parallel, using a hybrid approach combining global and local methods.
///
/// This function provides a parallel implementation of global optimization, which
/// can significantly speed up the optimization process for computationally
/// expensive problems. It uses `ParallelDifferentialEvolution` as the global
/// optimizer and `ParallelLevenbergMarquardt` for local refinement.
///
/// # Arguments
///
/// * `problem` - The problem to solve, implementing the `Problem` trait
/// * `bounds` - Lower and upper bounds for each parameter as Vec of (min, max) tuples
/// * `iterations` - Maximum number of iterations for global optimization
/// * `population_size` - Population size (for population-based methods)
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
///
/// # Example
///
/// ```rust,no_run
/// use lmopt_rs::{optimize_global_parallel, Problem};
/// use ndarray::Array1;
///
/// struct MyProblem;
/// impl Problem for MyProblem {
///    // Problem implementation...
///    # fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> { Ok(Array1::zeros(1)) }
///    # fn parameter_count(&self) -> usize { 1 }
///    # fn residual_count(&self) -> usize { 1 }
/// }
///
/// let problem = MyProblem;
/// let bounds = vec![(-10.0, 10.0)];
///
/// // Run parallel global optimization
/// let result = optimize_global_parallel(
///     &problem,    // Problem to solve
///     &bounds,     // Parameter bounds
///     100,         // Number of iterations
///     20,          // Population size
///     1e-6         // Tolerance
/// ).unwrap();
///
/// println!("Global minimum found at: {:.3}", result.params[0]);
/// println!("Function value: {:.6}", result.cost);
/// ```
pub fn optimize_global_parallel<P: Problem + Sync>(
    problem: &P,
    bounds: &[(f64, f64)],
    iterations: usize,
    population_size: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Create a parallel differential evolution optimizer
    let optimizer = ParallelDifferentialEvolution::new()
        .with_population_size(population_size)
        .with_local_optimization(true);

    // Run the optimization
    optimizer.optimize(problem, bounds, iterations, population_size, tol)
}

/// Run global optimization for a parameter-based problem.
///
/// This convenience function works with problems that implement the `ParameterProblem`
/// trait, which provides a higher-level interface for optimization problems with
/// named parameters, bounds, and constraints. It's particularly useful for fitting
/// models to data.
///
/// The function uses the `HybridGlobal` optimizer by default, which combines
/// global exploration with local refinement. Parameter bounds are automatically
/// extracted from the problem.
///
/// # Arguments
///
/// * `problem` - The parameter problem to solve, implementing `ParameterProblem` trait
/// * `iterations` - Maximum number of iterations for global optimization
/// * `population_size` - Population size (for population-based methods)
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
pub fn optimize_global_param_problem<P: ParameterProblem>(
    problem: &mut P,
    iterations: usize,
    population_size: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Get bounds from parameters
    let bounds = HybridGlobal::get_bounds_from_parameters(problem)?;

    // Create problem adapter
    let adapter = crate::problem_params::problem_from_parameter_problem(problem);

    // Get initial parameters
    let initial_params = problem.parameters_to_array()?;

    // Run global optimization
    let optimizer = HybridGlobal::new();
    let result = optimizer.optimize(&adapter, &bounds, iterations, population_size, tol)?;

    // Update parameters with optimized values
    problem.update_parameters_from_array(&result.params)?;

    Ok(result)
}

/// Generate a random point within the given bounds.
///
/// # Arguments
///
/// * `bounds` - Lower and upper bounds for each parameter
/// * `rng` - Random number generator
///
/// # Returns
///
/// * A random point within the bounds
fn random_point(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Array1<f64> {
    let point: Vec<f64> = bounds
        .iter()
        .map(|(min, max)| {
            if min.is_finite() && max.is_finite() {
                rng.gen_range(*min..*max)
            } else if min.is_finite() {
                min + rng.gen::<f64>() * 10.0
            } else if max.is_finite() {
                max - rng.gen::<f64>() * 10.0
            } else {
                rng.gen_range(-10.0..10.0)
            }
        })
        .collect();

    Array1::from_vec(point)
}

/// Calculate the cost (sum of squared residuals) for a point.
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `point` - The point to evaluate
///
/// # Returns
///
/// * The cost (sum of squared residuals)
fn calculate_cost<P: Problem>(problem: &P, point: &Array1<f64>) -> Result<f64> {
    // Evaluate the residuals
    let residuals = problem.eval(point)?;

    // Calculate the sum of squared residuals
    let cost = residuals.iter().map(|r| r.powi(2)).sum();

    Ok(cost)
}

/// Clip a point to the given bounds.
///
/// # Arguments
///
/// * `point` - The point to clip
/// * `bounds` - Lower and upper bounds for each parameter
///
/// # Returns
///
/// * The clipped point
fn clip_to_bounds(point: &Array1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
    let mut clipped = point.clone();

    for (i, (min, max)) in bounds.iter().enumerate() {
        if i < clipped.len() {
            if min.is_finite() && clipped[i] < *min {
                clipped[i] = *min;
            }
            if max.is_finite() && clipped[i] > *max {
                clipped[i] = *max;
            }
        }
    }

    clipped
}

/// Create a population of random points within the given bounds.
///
/// # Arguments
///
/// * `bounds` - Lower and upper bounds for each parameter
/// * `pop_size` - The size of the population
/// * `rng` - Random number generator
///
/// # Returns
///
/// * A population of random points
fn create_population(
    bounds: &[(f64, f64)],
    pop_size: usize,
    rng: &mut impl Rng,
) -> Vec<Array1<f64>> {
    (0..pop_size).map(|_| random_point(bounds, rng)).collect()
}

/// Evaluate the cost for each point in a population.
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `population` - The population to evaluate
///
/// # Returns
///
/// * A vector of costs for each point
fn evaluate_population<P: Problem>(problem: &P, population: &[Array1<f64>]) -> Result<Vec<f64>> {
    population
        .iter()
        .map(|point| calculate_cost(problem, point))
        .collect()
}
