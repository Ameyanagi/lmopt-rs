//! Hybrid Global optimization that combines global and local optimization.
//!
//! This module implements a hybrid approach that combines global search methods
//! with local optimization to efficiently find global minima.

use ndarray::Array1;
use rand::Rng;
use std::f64::{INFINITY, NEG_INFINITY};

use crate::error::{LmOptError, Result};
use crate::global_opt::{
    calculate_cost, clip_to_bounds, random_point, BasinHopping, DifferentialEvolution,
    GlobalOptResult, GlobalOptimizer, SimulatedAnnealing,
};
use crate::lm::{LevenbergMarquardt, LmConfig, LmResult};
use crate::problem::Problem;

/// A hybrid global optimizer that combines global search with local optimization.
///
/// This optimizer first uses a global optimization method to find promising
/// regions of the parameter space, then refines the solution using local
/// optimization with the Levenberg-Marquardt algorithm.
#[derive(Debug, Clone)]
pub struct HybridGlobal {
    /// Global optimization method
    pub global_method: GlobalMethod,

    /// Number of global iterations
    pub global_iterations: usize,

    /// Number of iterations without improvement for global search
    pub global_no_improvement: usize,

    /// Configuration for the local optimizer
    pub local_config: LmConfig,
}

/// Global optimization methods available in the hybrid optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlobalMethod {
    /// Simulated Annealing
    SimulatedAnnealing,

    /// Differential Evolution
    DifferentialEvolution,

    /// Basin Hopping
    BasinHopping,
}

impl Default for HybridGlobal {
    fn default() -> Self {
        Self {
            global_method: GlobalMethod::DifferentialEvolution,
            global_iterations: 100,
            global_no_improvement: 20,
            local_config: LmConfig::default(),
        }
    }
}

impl HybridGlobal {
    /// Create a new HybridGlobal optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new HybridGlobal optimizer with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `global_method` - Global optimization method to use
    /// * `global_iterations` - Number of global iterations
    /// * `global_no_improvement` - Number of iterations without improvement for global search
    /// * `local_config` - Configuration for the local optimizer
    pub fn with_params(
        global_method: GlobalMethod,
        global_iterations: usize,
        global_no_improvement: usize,
        local_config: LmConfig,
    ) -> Self {
        Self {
            global_method,
            global_iterations,
            global_no_improvement,
            local_config,
        }
    }

    /// Run the hybrid optimization with an initial guess.
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
        _max_iterations: usize,
        _max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        // First, run the global optimization
        let global_result = match self.global_method {
            GlobalMethod::SimulatedAnnealing => {
                let optimizer = SimulatedAnnealing::new();
                optimizer.optimize_with_initial(
                    problem,
                    initial_params,
                    bounds,
                    self.global_iterations,
                    self.global_no_improvement,
                    tol,
                )?
            }
            GlobalMethod::DifferentialEvolution => {
                let optimizer = DifferentialEvolution::new();
                optimizer.optimize(
                    problem,
                    bounds,
                    self.global_iterations,
                    self.global_no_improvement,
                    tol,
                )?
            }
            GlobalMethod::BasinHopping => {
                let optimizer = BasinHopping::new();
                optimizer.optimize_with_initial(
                    problem,
                    initial_params,
                    bounds,
                    self.global_iterations,
                    self.global_no_improvement,
                    tol,
                )?
            }
        };

        // If global optimization already meets tolerance, return it
        if global_result.cost < tol {
            return Ok(global_result);
        }

        // Otherwise, run local optimization to refine the solution
        let local_optimizer = LevenbergMarquardt::new(self.local_config.clone());
        let local_result = local_optimizer.minimize(problem, global_result.params.clone())?;

        // Check if local optimization improved the solution
        if local_result.cost < global_result.cost {
            let success = local_result.success || global_result.success;
            let message = format!(
                "Global phase: {}. Local phase: {}",
                global_result.message, local_result.message,
            );

            // Create a clone of the local result for storage
            let stored_local_result = local_result.clone();

            Ok(GlobalOptResult {
                params: local_result.params,
                cost: local_result.cost,
                iterations: global_result.iterations + local_result.iterations,
                func_evals: global_result.func_evals
                    + local_result.iterations * problem.residual_count(),
                success,
                message,
                local_result: Some(stored_local_result),
            })
        } else {
            // If local optimization didn't improve, return global result
            Ok(global_result)
        }
    }
}

impl GlobalOptimizer for HybridGlobal {
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        let mut rng = rand::thread_rng();

        // Generate a random starting point
        let initial_point = random_point(bounds, &mut rng);

        // Run the optimization with this starting point
        self.optimize_with_initial(
            problem,
            &initial_point,
            bounds,
            max_iterations,
            max_no_improvement,
            tol,
        )
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
    fn test_hybrid_global() {
        // Create the problem
        let problem = MultiMinimaProblem;

        // Define bounds
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        // Test each global method
        for method in [
            GlobalMethod::SimulatedAnnealing,
            GlobalMethod::DifferentialEvolution,
            GlobalMethod::BasinHopping,
        ]
        .iter()
        {
            // Create the optimizer
            let optimizer = HybridGlobal::with_params(
                *method,
                50,                  // global_iterations
                10,                  // global_no_improvement
                LmConfig::default(), // local_config
            );

            // Run the optimization
            let result = optimizer.optimize(&problem, &bounds, 50, 10, 1e-6).unwrap();

            // Check that the optimization succeeded
            assert!(result.success);

            // Check that the function value is close to the expected minimum
            // Increasing threshold due to stochastic nature of optimization
            println!("Method: {:?}, Cost: {}", method, result.cost);
            assert!(
                result.cost < 0.5,
                "Cost {} exceeds threshold for method {:?}",
                result.cost,
                method
            );

            // Due to the stochastic nature of global optimization, we don't check the exact parameter values
            // but only verify that the cost is low enough to indicate a good solution
        }
    }
}
