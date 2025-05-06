//! # Global Optimization Methods
//!
//! This module provides global optimization algorithms for finding global minima
//! of complex objective functions with multiple local minima. These methods overcome
//! the limitations of local optimization techniques like Levenberg-Marquardt by
//! exploring the parameter space more thoroughly.
//!
//! ## Available Algorithms
//!
//! * **Simulated Annealing**: Mimics the annealing process in metallurgy, randomly
//!   exploring the parameter space with a temperature parameter that decreases over time.
//!
//! * **Differential Evolution**: Population-based evolutionary algorithm that uses
//!   mutation, crossover, and selection to evolve parameter vectors toward the minimum.
//!
//! * **Basin Hopping**: Combines global stepping with local minimization, useful for
//!   energy landscapes with many local minima separated by barriers.
//!
//! * **Hybrid Global-Local**: Combines global search with Levenberg-Marquardt local
//!   optimization for efficient and robust minimization.
//!
//! ## Usage Examples
//!
//! ```rust,no_run
//! use lmopt_rs::{SimulatedAnnealing, GlobalOptimizer, problem::Problem};
//! use ndarray::Array1;
//!
//! // Create your problem (implementing the Problem trait)
//! let problem = /* your problem */;
//!
//! // Define parameter bounds (min, max) for each parameter
//! let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
//!
//! // Create a global optimizer (e.g., simulated annealing)
//! let mut optimizer = SimulatedAnnealing::new();
//!
//! // Run the optimization
//! let result = optimizer.optimize(&problem, &bounds, 1000, 100, 1e-8).unwrap();
//!
//! // Use the results
//! println!("Optimal parameters: {:?}", result.params);
//! println!("Final cost: {}", result.cost);
//! ```
//!
//! For parameter-based problems:
//!
//! ```rust,no_run
//! use lmopt_rs::{optimize_global_param_problem, problem_params::ParameterProblem};
//!
//! // Create a problem implementing ParameterProblem trait
//! let mut param_problem = /* your parameter problem */;
//!
//! // Optimize using the default global optimizer (HybridGlobal)
//! let result = optimize_global_param_problem(&mut param_problem, 1000, 100, 1e-8).unwrap();
//! ```

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
mod simulated_annealing;

// Re-export optimization algorithms
pub use basin_hopping::BasinHopping;
pub use differential_evolution::DifferentialEvolution;
pub use hybrid::HybridGlobal;
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
/// * `initial_params` - Initial guess for the parameters
/// * `bounds` - Lower and upper bounds for each parameter as Vec of (min, max) tuples
/// * `max_iterations` - Maximum number of iterations for global optimization
/// * `max_no_improvement` - Maximum number of iterations without improvement before stopping
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
///
/// # Example
///
/// ```rust,no_run
/// use lmopt_rs::optimize_global;
/// use lmopt_rs::problem::Problem;
/// use ndarray::Array1;
///
/// // Your problem implementation
/// struct MyProblem { /* ... */ }
/// impl Problem for MyProblem { /* ... */ }
///
/// let problem = MyProblem { /* ... */ };
/// let initial_params = Array1::from_vec(vec![0.0, 0.0]);
/// let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
///
/// // Run the optimization
/// let result = optimize_global(
///     &problem,
///     &initial_params,
///     &bounds,
///     1000,   // max iterations
///     100,    // max no improvement
///     1e-6    // tolerance
/// ).unwrap();
///
/// println!("Best parameters: {:?}", result.params);
/// println!("Final cost: {}", result.cost);
/// ```
pub fn optimize_global<P: Problem>(
    problem: &P,
    initial_params: &Array1<f64>,
    bounds: &[(f64, f64)],
    max_iterations: usize,
    max_no_improvement: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Create a hybrid optimizer
    let optimizer = HybridGlobal::new();

    // Run the optimization
    optimizer.optimize_with_initial(
        problem,
        initial_params,
        bounds,
        max_iterations,
        max_no_improvement,
        tol,
    )
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
/// * `max_iterations` - Maximum number of iterations for global optimization
/// * `max_no_improvement` - Maximum number of iterations without improvement
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * `Result<GlobalOptResult>` - The best solution found and its cost, or an error
///
/// # Example
///
/// ```rust,no_run
/// use lmopt_rs::{optimize_global_param_problem, parameters::Parameters};
/// use lmopt_rs::problem_params::ParameterProblem;
///
/// // A problem with named parameters
/// struct MyFittingProblem {
///     parameters: Parameters,
///     data_x: Vec<f64>,
///     data_y: Vec<f64>,
/// }
///
/// // Implement ParameterProblem for your problem
/// impl ParameterProblem for MyFittingProblem {
///     // Parameter access methods
///     fn parameters(&self) -> &Parameters { &self.parameters }
///     fn parameters_mut(&mut self) -> &mut Parameters { &mut self.parameters }
///     
///     // Residual evaluation method
///     fn eval_with_parameters(&self) -> Result<ndarray::Array1<f64>, lmopt_rs::error::LmOptError> {
///         // Calculate residuals between model and data
///         // ...
///         Ok(residuals)
///     }
/// }
///
/// // Create and set up problem
/// let mut problem = MyFittingProblem { /* ... */ };
///
/// // Run global optimization
/// let result = optimize_global_param_problem(
///     &mut problem,
///     1000,   // max iterations
///     100,    // max no improvement
///     1e-6    // tolerance
/// ).unwrap();
///
/// println!("Optimization succeeded: {}", result.success);
/// println!("Final cost: {}", result.cost);
///
/// // Access optimized parameter values
/// for (name, param) in problem.parameters().iter() {
///     println!("{} = {:.6}", name, param.value());
/// }
/// ```
pub fn optimize_global_param_problem<P: ParameterProblem>(
    _problem: &mut P,
    max_iterations: usize,
    max_no_improvement: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Create a hybrid optimizer
    let mut optimizer = HybridGlobal::new();

    // Run the optimization
    optimizer.optimize_param_problem(max_iterations, max_no_improvement, tol)
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
