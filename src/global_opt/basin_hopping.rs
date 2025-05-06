//! Basin Hopping algorithm for global optimization.
//!
//! This module implements the Basin Hopping algorithm, a stochastic algorithm
//! that combines global stepping with local minimization.

use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::f64::{INFINITY, NEG_INFINITY};

use crate::error::{LmOptError, Result};
use crate::global_opt::{
    calculate_cost, clip_to_bounds, random_point, GlobalOptResult, GlobalOptimizer,
};
use crate::lm::{LevenbergMarquardt, LmConfig, LmResult};
use crate::problem::Problem;

/// Basin Hopping algorithm for global optimization.
///
/// Basin Hopping is a stochastic algorithm that combines global stepping
/// with local minimization. It is effective for finding global minima in
/// continuous functions, especially those with many local minima separated
/// by significant barriers.
#[derive(Debug, Clone)]
pub struct BasinHopping {
    /// Temperature for Metropolis acceptance
    pub temperature: f64,

    /// Step size for perturbation
    pub step_size: f64,

    /// Number of initial points to try
    pub n_initial_points: usize,

    /// Local optimizer configuration
    pub local_config: LmConfig,
}

impl Default for BasinHopping {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            step_size: 0.5,
            n_initial_points: 5,
            local_config: LmConfig::default(),
        }
    }
}

impl BasinHopping {
    /// Create a new BasinHopping optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new BasinHopping optimizer with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature for Metropolis acceptance
    /// * `step_size` - Step size for perturbation
    /// * `n_initial_points` - Number of initial points to try
    /// * `local_config` - Local optimizer configuration
    pub fn with_params(
        temperature: f64,
        step_size: f64,
        n_initial_points: usize,
        local_config: LmConfig,
    ) -> Self {
        Self {
            temperature,
            step_size,
            n_initial_points,
            local_config,
        }
    }

    /// Run the Basin Hopping algorithm with an initial guess.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem to solve
    /// * `initial_params` - Initial guess for the parameters
    /// * `bounds` - Lower and upper bounds for each parameter
    /// * `max_iterations` - Maximum number of iterations
    /// * `max_no_improvement` - Maximum number of iterations without improvement
    /// * `tol` - Tolerance for convergence
    ///
    /// # Returns
    ///
    /// * The best solution found and its cost
    pub fn optimize_with_initial<P: Problem>(
        &self,
        problem: &P,
        initial_params: &Array1<f64>,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        let mut rng = rand::thread_rng();

        // Ensure bounds match parameter count
        if initial_params.len() != bounds.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} bounds for parameters, got {}",
                initial_params.len(),
                bounds.len()
            )));
        }

        // Create local optimizer
        let local_optimizer = LevenbergMarquardt::new(self.local_config.clone());

        // Run local optimization on the initial guess
        let mut current_params = initial_params.clone();
        let mut current_result = local_optimizer.minimize(problem, current_params.clone())?;
        current_params = current_result.params.clone();
        let mut current_cost = current_result.cost;

        // Initialize best solution
        let mut best_params = current_params.clone();
        let mut best_cost = current_cost;
        let mut best_result = current_result.clone();

        // Initialize counters
        let mut iterations = 0;
        let mut no_improvement = 0;
        let mut func_evals = problem.residual_count(); // Count initial evaluation

        // Run the optimization loop
        while iterations < max_iterations && no_improvement < max_no_improvement {
            // Perturb the current solution
            let candidate = self.perturb_solution(&current_params, bounds, &mut rng);

            // Run local optimization from the perturbed point
            let candidate_result = local_optimizer.minimize(problem, candidate)?;
            let candidate_params = candidate_result.params.clone();
            let candidate_cost = candidate_result.cost;

            // Update function evaluation count
            func_evals += candidate_result.iterations * problem.residual_count();

            // Calculate the acceptance probability using Metropolis criterion
            let accept = if candidate_cost <= current_cost {
                // Always accept better solutions
                true
            } else {
                // Accept worse solutions with a probability that depends on
                // the temperature and the cost difference
                let probability = (-(candidate_cost - current_cost) / self.temperature).exp();
                rng.gen::<f64>() < probability
            };

            // Update current solution
            if accept {
                current_params = candidate_params;
                current_cost = candidate_cost;
                current_result = candidate_result;

                // Update best solution if necessary
                if current_cost < best_cost {
                    best_params = current_params.clone();
                    best_cost = current_cost;
                    best_result = current_result.clone();
                    no_improvement = 0;
                } else {
                    no_improvement += 1;
                }
            } else {
                no_improvement += 1;
            }

            // Check for convergence
            if best_cost < tol {
                break;
            }

            iterations += 1;
        }

        // Create the result
        let success = (best_cost < tol) || (no_improvement < max_no_improvement);
        let message = if best_cost < tol {
            format!(
                "Converged to solution with cost {:.2e} < {:.2e}",
                best_cost, tol
            )
        } else if no_improvement >= max_no_improvement {
            format!(
                "Stopped after {} iterations without improvement",
                max_no_improvement
            )
        } else {
            format!("Reached maximum number of iterations: {}", max_iterations)
        };

        Ok(GlobalOptResult {
            params: best_params,
            cost: best_cost,
            iterations,
            func_evals,
            success,
            message,
            local_result: Some(best_result),
        })
    }

    /// Perturb a solution by adding random values to each parameter.
    ///
    /// # Arguments
    ///
    /// * `solution` - The solution to perturb
    /// * `bounds` - Lower and upper bounds for each parameter
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// * A new solution perturbed from the original
    fn perturb_solution(
        &self,
        solution: &Array1<f64>,
        bounds: &[(f64, f64)],
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        // Create a new solution by perturbing each parameter
        let mut new_solution = solution.clone();

        for i in 0..solution.len() {
            // Get the bounds for this parameter
            let (min, max) = bounds[i];

            // Calculate the parameter range
            let range = if min.is_finite() && max.is_finite() {
                max - min
            } else {
                // Use a default range if bounds are infinite
                10.0
            };

            // Calculate the step size as a fraction of the range
            let step = range * self.step_size;

            // Add a random perturbation
            let perturbation = Uniform::new(-step, step).sample(rng);
            new_solution[i] += perturbation;
        }

        // Clip the solution to the bounds
        clip_to_bounds(&new_solution, bounds)
    }
}

impl GlobalOptimizer for BasinHopping {
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        let mut rng = rand::thread_rng();

        // Generate multiple random starting points
        let mut best_result = None;
        let mut best_cost = INFINITY;

        // Try multiple starting points
        for _ in 0..self.n_initial_points {
            // Generate a random starting point
            let initial_point = random_point(bounds, &mut rng);

            // Run the optimization with this starting point
            let result = self.optimize_with_initial(
                problem,
                &initial_point,
                bounds,
                max_iterations,
                max_no_improvement,
                tol,
            )?;

            // Update the best result if this one is better
            if result.cost < best_cost {
                best_cost = result.cost;
                best_result = Some(result);
            }
        }

        // Return the best result
        best_result.ok_or_else(|| {
            LmOptError::ComputationError("Basin hopping failed to find a solution".to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Problem;
    use ndarray::array;

    /// A simple test problem with multiple local minima.
    struct MultiMinimaProblem;

    impl Problem for MultiMinimaProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            // Check parameter dimensions
            if params.len() != 2 {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected 2 parameters, got {}",
                    params.len()
                )));
            }

            let x = params[0];
            let y = params[1];

            // 2D function with multiple minima:
            // f(x, y) = sin(x) * cos(y) + 0.1 * x^2 + 0.1 * y^2
            // Global minimum at approximately (-1.57, 0) with value close to -1.0
            let f = (x.sin() * y.cos()) + 0.1 * x.powi(2) + 0.1 * y.powi(2);

            // Return the function value as a single-element residual
            Ok(array![f])
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_basin_hopping() {
        // Create the problem
        let problem = MultiMinimaProblem;

        // Define bounds
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        // Create the optimizer
        let optimizer = BasinHopping::with_params(
            1.0,                 // temperature
            0.5,                 // step_size
            3,                   // n_initial_points
            LmConfig::default(), // local_config
        );

        // Run the optimization
        let result = optimizer.optimize(&problem, &bounds, 20, 5, 1e-6).unwrap();

        // Check that the optimization succeeded
        assert!(result.success);

        // Check that the function value is close to the expected minimum
        assert!(result.cost < 0.1);

        // Due to the stochastic nature of basin hopping, we don't check the exact parameter values
        // but only verify that the cost is low enough to indicate a good solution
    }
}
