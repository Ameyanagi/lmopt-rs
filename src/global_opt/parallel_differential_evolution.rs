//! Parallel implementation of Differential Evolution.
//!
//! This module provides a parallel implementation of the Differential Evolution algorithm,
//! a population-based evolutionary algorithm for global optimization of complex problems.
//! The parallel implementation leverages multiple CPU cores for faster optimization,
//! particularly beneficial for computationally expensive objective functions.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fmt;

use crate::error::{LmOptError, Result};
use crate::lm::{LevenbergMarquardt, LmConfig, LmResult, ParallelLevenbergMarquardt};
use crate::problem::Problem;

use super::parallel::{
    calculate_cost, clip_to_bounds, create_population_parallel, evaluate_population_parallel,
};
use super::{GlobalOptResult, GlobalOptimizer};

/// Parallel implementation of the Differential Evolution algorithm.
///
/// Differential Evolution is a population-based evolutionary algorithm that
/// uses vector differences for mutation. This parallel implementation
/// evaluates the population and creates new populations in parallel, which
/// can significantly speed up the optimization process for computationally
/// expensive objective functions.
///
/// # Example
///
/// ```
/// use lmopt_rs::{Problem, ParallelDifferentialEvolution};
/// use ndarray::{array, Array1};
///
/// // Define a problem
/// struct MyProblem;
/// impl Problem for MyProblem {
///     // Implementation details...
///     # fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
///     #     let x = params[0];
///     #     let y = params[1];
///     #     Ok(array![(x*x + y*y).sqrt()])
///     # }
///     # fn parameter_count(&self) -> usize { 2 }
///     # fn residual_count(&self) -> usize { 1 }
/// }
///
/// // Create and configure the optimizer
/// let optimizer = ParallelDifferentialEvolution::new()
///     .with_population_size(50)
///     .with_crossover_probability(0.7)
///     .with_differential_weight(0.8)
///     .with_local_optimization(true);
///
/// // Run the optimization
/// let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
/// let problem = MyProblem;
/// let result = optimizer.optimize(&problem, &bounds, 100, 20, 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct ParallelDifferentialEvolution {
    /// Crossover probability (CR): Controls the probability of crossover
    crossover_probability: f64,

    /// Differential weight (F): Controls the amplification of differential variation
    differential_weight: f64,

    /// Population size: Number of individuals in the population
    population_size: usize,

    /// Whether to use local optimization with Levenberg-Marquardt
    use_local_optimization: bool,

    /// Maximum number of local optimization iterations
    local_optimization_iterations: usize,

    /// Random number generator seed
    seed: u64,
}

impl Default for ParallelDifferentialEvolution {
    fn default() -> Self {
        Self {
            crossover_probability: 0.9,
            differential_weight: 0.8,
            population_size: 50,
            use_local_optimization: false,
            local_optimization_iterations: 100,
            seed: 42,
        }
    }
}

impl ParallelDifferentialEvolution {
    /// Create a new ParallelDifferentialEvolution optimizer with default parameters.
    ///
    /// Default values:
    /// - Crossover probability (CR): 0.9
    /// - Differential weight (F): 0.8
    /// - Population size: 50
    /// - Local optimization: disabled
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the crossover probability (CR).
    ///
    /// CR controls the probability of crossover for each parameter.
    /// Higher values increase the number of parameters that are crossed over,
    /// leading to more exploratory behavior.
    ///
    /// # Arguments
    ///
    /// * `cr` - Crossover probability (0.0 to 1.0)
    pub fn with_crossover_probability(mut self, cr: f64) -> Self {
        self.crossover_probability = cr.clamp(0.0, 1.0);
        self
    }

    /// Set the differential weight (F).
    ///
    /// F controls the amplification of the differential variation.
    /// Higher values lead to more exploration, while lower values
    /// lead to more exploitation of the current good solutions.
    ///
    /// # Arguments
    ///
    /// * `f` - Differential weight (typically 0.5 to 1.0)
    pub fn with_differential_weight(mut self, f: f64) -> Self {
        self.differential_weight = f.max(0.0);
        self
    }

    /// Set the population size.
    ///
    /// Larger populations can explore more of the search space but require
    /// more function evaluations per iteration. A general rule of thumb is
    /// to use 10 times the number of parameters.
    ///
    /// # Arguments
    ///
    /// * `size` - Population size
    pub fn with_population_size(mut self, size: usize) -> Self {
        self.population_size = size.max(4); // Need at least 4 individuals
        self
    }

    /// Enable or disable local optimization.
    ///
    /// When enabled, the best solution found by differential evolution
    /// is further refined using the Levenberg-Marquardt algorithm.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to use local optimization
    pub fn with_local_optimization(mut self, enable: bool) -> Self {
        self.use_local_optimization = enable;
        self
    }

    /// Set the maximum number of iterations for local optimization.
    ///
    /// # Arguments
    ///
    /// * `iterations` - Maximum iterations for local optimization
    pub fn with_local_optimization_iterations(mut self, iterations: usize) -> Self {
        self.local_optimization_iterations = iterations;
        self
    }

    /// Set the random number generator seed.
    ///
    /// Setting the seed allows for reproducible optimization runs.
    ///
    /// # Arguments
    ///
    /// * `seed` - RNG seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Create a trial individual using differential evolution mutation and crossover.
    ///
    /// # Arguments
    ///
    /// * `target` - The target individual to be potentially replaced
    /// * `population` - The current population
    /// * `costs` - The costs of the current population
    /// * `bounds` - Parameter bounds
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// * The trial individual
    fn create_trial(
        &self,
        target_idx: usize,
        population: &[Array1<f64>],
        bounds: &[(f64, f64)],
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let target = &population[target_idx];
        let dim = target.len();

        // Randomly select three different individuals, all different from target
        let mut available_indices: Vec<usize> =
            (0..population.len()).filter(|&i| i != target_idx).collect();

        if available_indices.len() < 3 {
            // Not enough distinct individuals, just clone the target
            return target.clone();
        }

        available_indices.shuffle(rng);
        let a_idx = available_indices[0];
        let b_idx = available_indices[1];
        let c_idx = available_indices[2];

        let a = &population[a_idx];
        let b = &population[b_idx];
        let c = &population[c_idx];

        // Create the mutant vector: a + F * (b - c)
        let mut trial = target.clone();

        // Randomly select the parameter to ensure at least one gets crossed over
        let j_rand = rng.gen_range(0..dim);

        // Apply crossover
        for j in 0..dim {
            if j == j_rand || rng.gen::<f64>() < self.crossover_probability {
                // Apply mutation and crossover
                trial[j] = a[j] + self.differential_weight * (b[j] - c[j]);
            }
            // Otherwise, keep the parameter from the target
        }

        // Ensure trial is within bounds
        clip_to_bounds(&trial, bounds)
    }
}

impl GlobalOptimizer for ParallelDifferentialEvolution {
    fn optimize<P: Problem + Sync>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        let n_params = problem.parameter_count();

        // Check if bounds match the number of parameters
        if bounds.len() != n_params {
            return Err(LmOptError::DimensionMismatch(format!(
                "Number of bounds ({}) does not match number of parameters ({})",
                bounds.len(),
                n_params
            )));
        }

        // Use the provided population size or default to 10 times the number of parameters
        let pop_size = if self.population_size > 0 {
            self.population_size
        } else {
            10 * n_params
        }
        .max(4); // Minimum of 4 individuals needed

        // Initialize population in parallel
        let mut population = create_population_parallel(bounds, pop_size, self.seed);

        // Evaluate initial population in parallel
        let mut costs = evaluate_population_parallel(problem, &population)?;

        // Find the best solution in the initial population
        let (best_idx, &best_cost) = costs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| {
                LmOptError::OptimizationFailure("Failed to find best solution".to_string())
            })?;

        let mut best_params = population[best_idx].clone();
        let mut best_cost = best_cost;

        // Initialize tracking variables
        let mut iterations = 0;
        let mut func_evals = pop_size;
        let mut no_improvement_count = 0;

        // Main optimization loop
        while iterations < max_iterations && no_improvement_count < max_no_improvement {
            // Create a new generation of individuals in parallel
            let mut rng_master = StdRng::seed_from_u64(self.seed.wrapping_add(iterations as u64));

            // Generate trial vectors in parallel
            let trials: Vec<_> = (0..pop_size)
                .into_par_iter()
                .map(|i| {
                    // Create a new RNG for this thread with a unique seed
                    let mut thread_rng = StdRng::seed_from_u64(
                        self.seed
                            .wrapping_add(iterations as u64)
                            .wrapping_add(i as u64),
                    );
                    self.create_trial(i, &population, bounds, &mut thread_rng)
                })
                .collect();

            // Evaluate trial vectors in parallel
            let trial_costs = evaluate_population_parallel(problem, &trials)?;
            func_evals += pop_size;

            // Selection: keep the better of target and trial
            let mut improved = false;

            // Process selection and combine results (serial version for now)
            let selection_results: Vec<_> = costs
                .iter()
                .zip(trial_costs.iter())
                .enumerate()
                .map(|(i, (&old_cost, &new_cost))| {
                    if new_cost <= old_cost {
                        // Trial is better than or equal to target
                        (true, new_cost, i)
                    } else {
                        // Target is better
                        (false, old_cost, i)
                    }
                })
                .collect();

            // Apply selection results
            for &(replace, cost, i) in &selection_results {
                if replace {
                    population[i] = trials[i].clone();
                    costs[i] = cost;

                    // Update best solution if this is better
                    if cost < best_cost {
                        best_cost = cost;
                        best_params = population[i].clone();
                        improved = true;
                    }
                }
            }

            // Update iteration count
            iterations += 1;

            // Update no improvement counter
            if improved {
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            // Check for convergence
            if best_cost < tol {
                break;
            }
        }

        // Local optimization (optional)
        let local_result = if self.use_local_optimization {
            // Use parallel LM algorithm for local refinement
            let lm = ParallelLevenbergMarquardt::new()
                .with_max_iterations(self.local_optimization_iterations);

            match lm.minimize(problem, best_params.clone()) {
                Ok(result) => {
                    // Update best solution if local optimization improved it
                    if result.cost < best_cost {
                        best_params = result.params.clone();
                        best_cost = result.cost;
                    }
                    func_evals += result.func_evals;
                    Some(result)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        // Create success message
        let success = best_cost < tol || iterations < max_iterations;
        let message = if best_cost < tol {
            format!("Optimization converged to tolerance {:.2e}", tol)
        } else if iterations >= max_iterations {
            format!("Reached maximum iterations ({})", max_iterations)
        } else {
            format!(
                "Reached maximum iterations without improvement ({})",
                max_no_improvement
            )
        };

        // Return the result
        Ok(GlobalOptResult {
            params: best_params,
            cost: best_cost,
            iterations,
            func_evals,
            success,
            message,
            local_result,
        })
    }
}

impl fmt::Display for ParallelDifferentialEvolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Parallel Differential Evolution:")?;
        writeln!(
            f,
            "  Crossover probability (CR): {:.4}",
            self.crossover_probability
        )?;
        writeln!(
            f,
            "  Differential weight (F): {:.4}",
            self.differential_weight
        )?;
        writeln!(f, "  Population size: {}", self.population_size)?;
        writeln!(f, "  Local optimization: {}", self.use_local_optimization)?;
        if self.use_local_optimization {
            writeln!(
                f,
                "  Local optimization iterations: {}",
                self.local_optimization_iterations
            )?;
        }
        writeln!(f, "  Random seed: {}", self.seed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_opt::differential_evolution::DifferentialEvolution;
    use ndarray::array;
    use std::time::Instant;

    // Simple test problem with known minimum at (0, 0)
    struct SphereProblem;

    impl Problem for SphereProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            let sum_of_squares = params.iter().map(|&x| x * x).sum::<f64>();
            Ok(array![sum_of_squares.sqrt()])
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    // Rosenbrock function with known minimum at (1, 1)
    struct RosenbrockProblem;

    impl Problem for RosenbrockProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            if params.len() != 2 {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected 2 parameters, got {}",
                    params.len()
                )));
            }

            let x = params[0];
            let y = params[1];

            // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
            let term1 = (1.0 - x).powi(2);
            let term2 = 100.0 * (y - x.powi(2)).powi(2);

            Ok(array![(term1 + term2).sqrt()])
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_parallel_sphere_optimization() {
        let problem = SphereProblem;
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        // Configure the optimizer
        let optimizer = ParallelDifferentialEvolution::new()
            .with_population_size(40)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_seed(42);

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &bounds, 100, 20, 1e-6)
            .unwrap();

        // Check that we found a solution close to the known minimum (0, 0)
        println!("Optimized parameters: {:?}", result.params);
        println!("Cost: {}", result.cost);

        assert!(result.params[0].abs() < 0.1, "x parameter not close to 0");
        assert!(result.params[1].abs() < 0.1, "y parameter not close to 0");
        assert!(result.cost < 0.01, "Cost not minimized sufficiently");
    }

    #[test]
    fn test_parallel_rosenbrock_optimization() {
        let problem = RosenbrockProblem;
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        // Configure the optimizer with local optimization
        let optimizer = ParallelDifferentialEvolution::new()
            .with_population_size(40)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_local_optimization(true)
            .with_local_optimization_iterations(100)
            .with_seed(42);

        // Run the optimization
        let result = optimizer
            .optimize(&problem, &bounds, 100, 20, 1e-6)
            .unwrap();

        // Check that we found a solution close to the known minimum (1, 1)
        println!("Optimized parameters: {:?}", result.params);
        println!("Cost: {}", result.cost);

        assert!(
            (result.params[0] - 1.0).abs() < 0.1,
            "x parameter not close to 1"
        );
        assert!(
            (result.params[1] - 1.0).abs() < 0.1,
            "y parameter not close to 1"
        );
        assert!(result.cost < 0.01, "Cost not minimized sufficiently");
    }

    #[test]
    fn compare_parallel_vs_sequential() {
        // Create a more complex, higher-dimensional problem
        struct HighDimProblem {
            dimension: usize,
        }

        impl HighDimProblem {
            fn new(dimension: usize) -> Self {
                Self { dimension }
            }
        }

        impl Problem for HighDimProblem {
            fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
                if params.len() != self.dimension {
                    return Err(LmOptError::DimensionMismatch(format!(
                        "Expected {} parameters, got {}",
                        self.dimension,
                        params.len()
                    )));
                }

                let mut sum = 0.0;
                for x in params.iter() {
                    // Simulate more expensive calculation
                    let mut term = 0.0;
                    for i in 1..10 {
                        term += (i as f64 * x).sin() / (i as f64);
                    }
                    sum += x * x + term * term;
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

        // Problem with 20 dimensions
        let problem = HighDimProblem::new(20);

        // Bounds for all parameters
        let bounds = vec![(-5.0, 5.0); 20];

        // Sequential optimization
        let sequential_optimizer = DifferentialEvolution::new()
            .with_population_size(40)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_seed(42);

        let start_time = Instant::now();
        let sequential_result = sequential_optimizer
            .optimize(&problem, &bounds, 20, 10, 1e-4)
            .unwrap();
        let sequential_time = start_time.elapsed();

        // Parallel optimization
        let parallel_optimizer = ParallelDifferentialEvolution::new()
            .with_population_size(40)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_seed(42);

        let start_time = Instant::now();
        let parallel_result = parallel_optimizer
            .optimize(&problem, &bounds, 20, 10, 1e-4)
            .unwrap();
        let parallel_time = start_time.elapsed();

        // Print timing results
        println!("Sequential optimization time: {:?}", sequential_time);
        println!("Sequential iterations: {}", sequential_result.iterations);
        println!("Sequential cost: {}", sequential_result.cost);

        println!("Parallel optimization time: {:?}", parallel_time);
        println!("Parallel iterations: {}", parallel_result.iterations);
        println!("Parallel cost: {}", parallel_result.cost);

        println!(
            "Speedup: {:.2}x",
            sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
        );

        // Both should converge to similar cost values
        let cost_ratio = sequential_result.cost / parallel_result.cost;
        assert!(
            cost_ratio > 0.5 && cost_ratio < 2.0,
            "Sequential and parallel optimizers found significantly different solutions"
        );
    }
}
