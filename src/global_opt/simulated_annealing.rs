//! Simulated Annealing algorithm for global optimization.
//!
//! This module implements the Simulated Annealing algorithm, a probabilistic
//! technique for finding global minima in complex search spaces.

use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::f64::{INFINITY, NEG_INFINITY};

use crate::error::{LmOptError, Result};
use crate::global_opt::{
    calculate_cost, clip_to_bounds, random_point, GlobalOptResult, GlobalOptimizer,
};
use crate::problem::Problem;

/// Simulated Annealing algorithm for global optimization.
///
/// Simulated Annealing is a probabilistic technique for finding global minima
/// in complex search spaces. It is inspired by the process of annealing in metallurgy.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing {
    /// Initial temperature
    pub initial_temp: f64,

    /// Cooling rate
    pub cooling_rate: f64,

    /// Population size
    pub pop_size: usize,

    /// Step size
    pub step_size: f64,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self {
            initial_temp: 100.0,
            cooling_rate: 0.95,
            pop_size: 10,
            step_size: 0.1,
        }
    }
}

impl SimulatedAnnealing {
    /// Create a new SimulatedAnnealing optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new SimulatedAnnealing optimizer with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `initial_temp` - Initial temperature
    /// * `cooling_rate` - Cooling rate (0.9-0.99 is typical)
    /// * `pop_size` - Population size
    /// * `step_size` - Step size for perturbations
    pub fn with_params(
        initial_temp: f64,
        cooling_rate: f64,
        pop_size: usize,
        step_size: f64,
    ) -> Self {
        Self {
            initial_temp,
            cooling_rate,
            pop_size,
            step_size,
        }
    }

    /// Run the Simulated Annealing algorithm with an initial guess.
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

        // Initialize best solution with the initial guess
        let mut current_solution = initial_params.clone();
        let mut current_cost = calculate_cost(problem, &current_solution)?;

        let mut best_solution = current_solution.clone();
        let mut best_cost = current_cost;

        // Initialize temperature
        let mut temperature = self.initial_temp;

        // Initialize counters
        let mut iterations = 0;
        let mut no_improvement = 0;
        let mut func_evals = 1; // Count initial evaluation

        // Run the optimization loop
        while iterations < max_iterations && no_improvement < max_no_improvement {
            // Generate a new candidate solution by perturbing the current solution
            let candidate = self.perturb_solution(&current_solution, bounds, &mut rng);

            // Evaluate the candidate
            let candidate_cost = calculate_cost(problem, &candidate)?;
            func_evals += 1;

            // Calculate the difference in cost
            let cost_diff = candidate_cost - current_cost;

            // Determine whether to accept the candidate
            let accept = if cost_diff <= 0.0 {
                // If the candidate is better, always accept it
                true
            } else {
                // If the candidate is worse, accept it with a probability that depends on
                // the current temperature and the cost difference
                let probability = (-cost_diff / temperature).exp();
                rng.gen::<f64>() < probability
            };

            // Update current solution
            if accept {
                current_solution = candidate;
                current_cost = candidate_cost;

                // Update best solution if necessary
                if current_cost < best_cost {
                    best_solution = current_solution.clone();
                    best_cost = current_cost;
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

            // Cool the temperature
            temperature *= self.cooling_rate;
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
            params: best_solution,
            cost: best_cost,
            iterations,
            func_evals,
            success,
            message,
            local_result: None,
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

impl GlobalOptimizer for SimulatedAnnealing {
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
        for _ in 0..self.pop_size {
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
            LmOptError::ComputationError(
                "Simulated annealing failed to find a solution".to_string(),
            )
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
    fn test_simulated_annealing() {
        // Create the problem
        let problem = MultiMinimaProblem;

        // Define bounds
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        // Create the optimizer
        let optimizer = SimulatedAnnealing::with_params(100.0, 0.95, 5, 0.1);

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &bounds, 1000, 100, 1e-6)
            .unwrap();

        // Due to the stochastic nature of simulated annealing, the success flag may vary
        // We only check that the cost is reasonable

        // Check that the function value is close to the expected minimum
        assert!(result.cost < 0.1);

        // Due to the stochastic nature of simulated annealing, we don't check the exact parameter values
        // but only verify that the cost is low enough to indicate a good solution
    }
}
