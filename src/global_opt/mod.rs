//! Global optimization methods for finding global minima.
//!
//! This module provides global optimization algorithms that can be used to find
//! global minima of complex objective functions with multiple local minima. These
//! methods are particularly useful for fitting models with many parameters.

use ndarray::{Array1, Array2};
use rand::Rng;
use std::fmt;

use crate::error::{LmOptError, Result};
use crate::problem::Problem;
use crate::problem_params::ParameterProblem;
use crate::lm::{LevenbergMarquardt, LmConfig, LmResult};

/// Trait for global optimization methods.
pub trait GlobalOptimizer {
    /// Run the optimization and return the best solution found.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem to solve
    /// * `bounds` - Lower and upper bounds for each parameter
    /// * `max_iterations` - Maximum number of iterations
    /// * `max_no_improvement` - Maximum number of iterations without improvement
    /// * `tol` - Tolerance for convergence
    ///
    /// # Returns
    ///
    /// * The best solution found and its cost
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
            "optimize_param_problem not implemented for this optimizer".to_string()
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
            let effective_min = if min == f64::NEG_INFINITY { f64::NEG_INFINITY } else { min };
            let effective_max = if max == f64::INFINITY { f64::INFINITY } else { max };
            
            bounds.push((effective_min, effective_max));
        }
        
        Ok(bounds)
    }
}

/// Result of a global optimization.
#[derive(Debug, Clone)]
pub struct GlobalOptResult {
    /// The best parameters found
    pub params: Array1<f64>,
    
    /// The best cost found
    pub cost: f64,
    
    /// The number of iterations performed
    pub iterations: usize,
    
    /// The number of function evaluations
    pub func_evals: usize,
    
    /// Whether the optimization succeeded
    pub success: bool,
    
    /// A message describing the result
    pub message: String,
    
    /// Local optimization result if a local optimizer was used
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
mod simulated_annealing;
mod differential_evolution;
mod basin_hopping;
mod hybrid;

// Re-export optimization algorithms
pub use simulated_annealing::SimulatedAnnealing;
pub use differential_evolution::DifferentialEvolution;
pub use basin_hopping::BasinHopping;
pub use hybrid::HybridGlobal;

/// Run global optimization using simulated annealing followed by
/// local optimization with Levenberg-Marquardt.
///
/// # Arguments
///
/// * `problem` - The problem to solve
/// * `initial_params` - Initial guess for the parameters
/// * `bounds` - Lower and upper bounds for each parameter
/// * `max_iterations` - Maximum number of iterations for global optimization
/// * `max_no_improvement` - Maximum number of iterations without improvement
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * The best solution found
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

/// Run global optimization for a parameter problem.
///
/// # Arguments
///
/// * `problem` - The parameter problem to solve
/// * `max_iterations` - Maximum number of iterations for global optimization
/// * `max_no_improvement` - Maximum number of iterations without improvement
/// * `tol` - Tolerance for convergence
///
/// # Returns
///
/// * The best solution found
pub fn optimize_global_param_problem<P: ParameterProblem>(
    problem: &mut P,
    max_iterations: usize,
    max_no_improvement: usize,
    tol: f64,
) -> Result<GlobalOptResult> {
    // Create a hybrid optimizer
    let mut optimizer = HybridGlobal::new();
    
    // Run the optimization
    optimizer.optimize_param_problem(
        max_iterations,
        max_no_improvement,
        tol,
    )
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
    (0..pop_size)
        .map(|_| random_point(bounds, rng))
        .collect()
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
fn evaluate_population<P: Problem>(
    problem: &P,
    population: &[Array1<f64>],
) -> Result<Vec<f64>> {
    population
        .iter()
        .map(|point| calculate_cost(problem, point))
        .collect()
}