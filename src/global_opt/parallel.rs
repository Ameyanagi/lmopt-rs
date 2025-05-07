//! Parallel implementations for global optimization.
//!
//! This module provides parallel versions of computationally intensive operations
//! used in global optimization, such as population evaluation and generation.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::error::Result;
use crate::problem::Problem;

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
pub fn calculate_cost<P: Problem + Sync>(problem: &P, point: &Array1<f64>) -> Result<f64> {
    // Evaluate the residuals
    let residuals = problem.eval(point)?;

    // Calculate the sum of squared residuals
    let cost = residuals.iter().map(|r| r.powi(2)).sum();

    Ok(cost)
}

/// Evaluate the cost for each point in a population in parallel.
///
/// This function uses Rayon to evaluate the cost of each point in the population
/// in parallel, which can significantly speed up the optimization process for
/// expensive objective functions.
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `population` - The population to evaluate
///
/// # Returns
///
/// * A vector of costs for each point
pub fn evaluate_population_parallel<P: Problem + Sync>(
    problem: &P,
    population: &[Array1<f64>],
) -> Result<Vec<f64>> {
    // Use Rayon's parallel iterator to evaluate the cost of each point in parallel
    population
        .par_iter()
        .map(|point| calculate_cost(problem, point))
        .collect()
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
pub fn random_point(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Array1<f64> {
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

/// Create a population of random points within the given bounds in parallel.
///
/// This function creates a population of random points within the given bounds
/// in parallel, using multiple threads to speed up the process.
///
/// # Arguments
///
/// * `bounds` - Lower and upper bounds for each parameter
/// * `pop_size` - The size of the population
/// * `seed` - Random number generator seed
///
/// # Returns
///
/// * A population of random points
pub fn create_population_parallel(
    bounds: &[(f64, f64)],
    pop_size: usize,
    seed: u64,
) -> Vec<Array1<f64>> {
    // Create a vector of indices for the population
    let indices: Vec<usize> = (0..pop_size).collect();

    // Use Rayon's parallel iterator to create the population in parallel
    // Each thread gets its own RNG seeded deterministically based on the index
    indices
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            random_point(bounds, &mut rng)
        })
        .collect()
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
pub fn clip_to_bounds(point: &Array1<f64>, bounds: &[(f64, f64)]) -> Array1<f64> {
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

/// Clip a population of points to the given bounds in parallel.
///
/// # Arguments
///
/// * `population` - The population to clip
/// * `bounds` - Lower and upper bounds for each parameter
///
/// # Returns
///
/// * The clipped population
pub fn clip_population_parallel(
    population: &[Array1<f64>],
    bounds: &[(f64, f64)],
) -> Vec<Array1<f64>> {
    // Use Rayon's parallel iterator to clip each point in parallel
    population
        .par_iter()
        .map(|point| clip_to_bounds(point, bounds))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LmOptError;
    use crate::global_opt;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::Instant;

    // Simple test problem
    struct TestProblem;

    impl Problem for TestProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            if params.len() != 2 {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected 2 parameters, got {}",
                    params.len()
                )));
            }

            let x = params[0];
            let y = params[1];

            // Simple function with minimum at (0, 0)
            // f(x, y) = x^2 + y^2
            Ok(array![(x.powi(2) + y.powi(2)).sqrt()])
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    #[test]
    fn test_parallel_vs_sequential_evaluate_population() {
        // Create a test problem
        let problem = TestProblem;

        // Create a population of random points
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let mut rng = StdRng::seed_from_u64(42);
        let large_population: Vec<Array1<f64>> = (0..1000)
            .map(|_| global_opt::random_point(&bounds, &mut rng))
            .collect();

        // Measure the time to evaluate the population sequentially
        let start_time = Instant::now();
        let sequential_costs =
            global_opt::evaluate_population(&problem, &large_population).unwrap();
        let sequential_time = start_time.elapsed();

        // Measure the time to evaluate the population in parallel
        let start_time = Instant::now();
        let parallel_costs = evaluate_population_parallel(&problem, &large_population).unwrap();
        let parallel_time = start_time.elapsed();

        // Verify that the results are the same
        for (i, (seq, par)) in sequential_costs
            .iter()
            .zip(parallel_costs.iter())
            .enumerate()
        {
            assert_eq!(seq, par, "Costs differ at index {}", i);
        }

        // Print the timing results
        println!("Sequential evaluation time: {:?}", sequential_time);
        println!("Parallel evaluation time: {:?}", parallel_time);
        println!(
            "Speedup: {:.2}x",
            sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
        );
    }

    #[test]
    fn test_parallel_population_creation() {
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
        let pop_size = 1000;
        let seed = 42;

        // Measure the time to create a population sequentially
        let start_time = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed);
        let sequential_pop = (0..pop_size)
            .map(|_| global_opt::random_point(&bounds, &mut rng))
            .collect::<Vec<Array1<f64>>>();
        let sequential_time = start_time.elapsed();

        // Measure the time to create a population in parallel
        let start_time = Instant::now();
        let parallel_pop = create_population_parallel(&bounds, pop_size, seed);
        let parallel_time = start_time.elapsed();

        // Verify the populations have the same size
        assert_eq!(sequential_pop.len(), parallel_pop.len());

        // Print the timing results
        println!("Sequential population creation time: {:?}", sequential_time);
        println!("Parallel population creation time: {:?}", parallel_time);
        println!(
            "Speedup: {:.2}x",
            sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
        );
    }
}
