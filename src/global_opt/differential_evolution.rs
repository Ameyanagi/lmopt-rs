//! Differential Evolution algorithm for global optimization.
//!
//! This module implements the Differential Evolution algorithm, a population-based
//! stochastic function minimizer that is particularly effective for global optimization.

use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::Rng;
use std::f64::{INFINITY, NEG_INFINITY};

use crate::error::{LmOptError, Result};
use crate::global_opt::{
    calculate_cost, clip_to_bounds, create_population, evaluate_population, random_point,
    GlobalOptResult, GlobalOptimizer,
};
use crate::problem::Problem;

/// Differential Evolution algorithm for global optimization.
///
/// Differential Evolution is a population-based stochastic function minimizer
/// that is effective for global optimization. It uses vector differences to
/// perturb the population.
#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    /// Population size multiplier (population size = multiplier * parameter count)
    pub pop_size_multiplier: usize,

    /// Differential weight (F) in range [0, 2]
    pub differential_weight: f64,

    /// Crossover probability (CR) in range [0, 1]
    pub crossover_prob: f64,

    /// Strategy for creating candidate solutions
    pub strategy: DEStrategy,
}

/// Strategies for creating candidate solutions in Differential Evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DEStrategy {
    /// DE/rand/1: x_r1 + F * (x_r2 - x_r3)
    Rand1,

    /// DE/rand/2: x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    Rand2,

    /// DE/best/1: x_best + F * (x_r1 - x_r2)
    Best1,

    /// DE/best/2: x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
    Best2,

    /// DE/current-to-best/1: x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    CurrentToBest1,
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self {
            pop_size_multiplier: 10,
            differential_weight: 0.8,
            crossover_prob: 0.9,
            strategy: DEStrategy::Rand1,
        }
    }
}

impl DifferentialEvolution {
    /// Create a new DifferentialEvolution optimizer with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new DifferentialEvolution optimizer with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `pop_size_multiplier` - Population size multiplier (population size = multiplier * parameter count)
    /// * `differential_weight` - Differential weight (F) in range [0, 2]
    /// * `crossover_prob` - Crossover probability (CR) in range [0, 1]
    /// * `strategy` - Strategy for creating candidate solutions
    pub fn with_params(
        pop_size_multiplier: usize,
        differential_weight: f64,
        crossover_prob: f64,
        strategy: DEStrategy,
    ) -> Self {
        Self {
            pop_size_multiplier,
            differential_weight,
            crossover_prob,
            strategy,
        }
    }

    /// Set the population size multiplier.
    ///
    /// # Arguments
    ///
    /// * `size` - Population size (not a multiplier)
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.pop_size_multiplier = size;
        self
    }

    /// Set the differential weight (F).
    ///
    /// # Arguments
    ///
    /// * `weight` - Differential weight in range [0, 2]
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_differential_weight(mut self, weight: f64) -> Self {
        self.differential_weight = weight;
        self
    }

    /// Set the crossover probability (CR).
    ///
    /// # Arguments
    ///
    /// * `prob` - Crossover probability in range [0, 1]
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_crossover_probability(mut self, prob: f64) -> Self {
        self.crossover_prob = prob;
        self
    }

    /// Set the random seed for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// * Self for method chaining
    pub fn with_seed(self, _seed: u64) -> Self {
        // In the current implementation, we don't actually use the seed,
        // but we provide this method for API compatibility with ParallelDifferentialEvolution
        self
    }

    /// Create a trial vector using the specified strategy.
    ///
    /// # Arguments
    ///
    /// * `target_idx` - Index of the target vector
    /// * `population` - Current population
    /// * `costs` - Costs for the current population
    /// * `bounds` - Parameter bounds
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// * A new trial vector
    fn create_trial_vector(
        &self,
        target_idx: usize,
        population: &[Array1<f64>],
        costs: &[f64],
        bounds: &[(f64, f64)],
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let n_params = population[0].len();
        let n_pop = population.len();

        // Find the best solution in the population
        let best_idx = costs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        // Create a trial vector based on the strategy
        let mut trial = match self.strategy {
            DEStrategy::Rand1 => {
                // Select 3 random vectors different from the target
                let mut available_indices: Vec<usize> =
                    (0..n_pop).filter(|&i| i != target_idx).collect();
                available_indices.shuffle(rng);

                let r1 = available_indices[0];
                let r2 = available_indices[1];
                let r3 = available_indices[2];

                // Create trial vector: x_r1 + F * (x_r2 - x_r3)
                let mut trial = population[r1].clone();
                for i in 0..n_params {
                    trial[i] += self.differential_weight * (population[r2][i] - population[r3][i]);
                }

                trial
            }
            DEStrategy::Rand2 => {
                // Select 5 random vectors different from the target
                let mut available_indices: Vec<usize> =
                    (0..n_pop).filter(|&i| i != target_idx).collect();
                available_indices.shuffle(rng);

                let r1 = available_indices[0];
                let r2 = available_indices[1];
                let r3 = available_indices[2];
                let r4 = available_indices[3];
                let r5 = available_indices[4];

                // Create trial vector: x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
                let mut trial = population[r1].clone();
                for i in 0..n_params {
                    trial[i] += self.differential_weight * (population[r2][i] - population[r3][i]);
                    trial[i] += self.differential_weight * (population[r4][i] - population[r5][i]);
                }

                trial
            }
            DEStrategy::Best1 => {
                // Select 2 random vectors different from the target and best
                let mut available_indices: Vec<usize> = (0..n_pop)
                    .filter(|&i| i != target_idx && i != best_idx)
                    .collect();
                available_indices.shuffle(rng);

                let r1 = available_indices[0];
                let r2 = available_indices[1];

                // Create trial vector: x_best + F * (x_r1 - x_r2)
                let mut trial = population[best_idx].clone();
                for i in 0..n_params {
                    trial[i] += self.differential_weight * (population[r1][i] - population[r2][i]);
                }

                trial
            }
            DEStrategy::Best2 => {
                // Select 4 random vectors different from the target and best
                let mut available_indices: Vec<usize> = (0..n_pop)
                    .filter(|&i| i != target_idx && i != best_idx)
                    .collect();
                available_indices.shuffle(rng);

                let r1 = available_indices[0];
                let r2 = available_indices[1];
                let r3 = available_indices[2];
                let r4 = available_indices[3];

                // Create trial vector: x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
                let mut trial = population[best_idx].clone();
                for i in 0..n_params {
                    trial[i] += self.differential_weight * (population[r1][i] - population[r2][i]);
                    trial[i] += self.differential_weight * (population[r3][i] - population[r4][i]);
                }

                trial
            }
            DEStrategy::CurrentToBest1 => {
                // Select 2 random vectors different from the target and best
                let mut available_indices: Vec<usize> = (0..n_pop)
                    .filter(|&i| i != target_idx && i != best_idx)
                    .collect();
                available_indices.shuffle(rng);

                let r1 = available_indices[0];
                let r2 = available_indices[1];

                // Create trial vector: x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
                let mut trial = population[target_idx].clone();
                for i in 0..n_params {
                    trial[i] += self.differential_weight
                        * (population[best_idx][i] - population[target_idx][i]);
                    trial[i] += self.differential_weight * (population[r1][i] - population[r2][i]);
                }

                trial
            }
        };

        // Apply crossover with the target vector
        let target = &population[target_idx];
        let j_rand = rng.gen_range(0..n_params);

        for j in 0..n_params {
            if rng.gen::<f64>() > self.crossover_prob && j != j_rand {
                trial[j] = target[j];
            }
        }

        // Clip to bounds
        clip_to_bounds(&trial, bounds)
    }
}

impl GlobalOptimizer for DifferentialEvolution {
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        let mut rng = rand::thread_rng();

        // Ensure bounds match parameter count
        let n_params = problem.parameter_count();
        if n_params != bounds.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} bounds for parameters, got {}",
                n_params,
                bounds.len()
            )));
        }

        // Create initial population
        let pop_size = self.pop_size_multiplier * n_params.max(4);
        let mut population = create_population(bounds, pop_size, &mut rng);

        // Evaluate initial population
        let mut costs = evaluate_population(problem, &population)?;

        // Find best solution
        let best_result = costs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let mut best_solution = population[best_result.0].clone();
        let mut best_cost = *best_result.1;

        // Initialize counters
        let mut iterations = 0;
        let mut no_improvement = 0;
        let mut func_evals = population.len();

        // Run the optimization loop
        while iterations < max_iterations && no_improvement < max_no_improvement {
            let mut improved = false;

            // Iterate over each member of the population
            for i in 0..pop_size {
                // Create a trial vector
                let trial = self.create_trial_vector(i, &population, &costs, bounds, &mut rng);

                // Evaluate the trial vector
                let trial_cost = calculate_cost(problem, &trial)?;
                func_evals += 1;

                // If the trial is better, replace the target vector
                if trial_cost < costs[i] {
                    population[i] = trial;
                    costs[i] = trial_cost;

                    // Update best solution if necessary
                    if trial_cost < best_cost {
                        best_solution = population[i].clone();
                        best_cost = trial_cost;
                        improved = true;
                    }
                }
            }

            // Check for improvement
            if improved {
                no_improvement = 0;
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
            params: best_solution,
            cost: best_cost,
            iterations,
            func_evals,
            success,
            message,
            local_result: None,
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
    fn test_differential_evolution() {
        // Create the problem
        let problem = MultiMinimaProblem;

        // Define bounds
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        // Create the optimizer
        let optimizer = DifferentialEvolution::with_params(
            5,                 // pop_size_multiplier
            0.8,               // differential_weight
            0.9,               // crossover_prob
            DEStrategy::Best1, // strategy
        );

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &bounds, 100, 10, 1e-6)
            .unwrap();

        // Check that the optimization succeeded
        assert!(result.success);

        // Check that the function value (cost) is close to the global minimum
        // Due to the stochastic nature of DE, we don't check the exact parameter values
        // but only verify that the cost is low enough to indicate a good solution
        assert!(result.cost < 0.1);
    }
}
