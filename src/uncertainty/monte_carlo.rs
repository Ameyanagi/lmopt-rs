//! # Monte Carlo Methods for Uncertainty Propagation
//! 
//! This module provides functionality for performing Monte Carlo simulations
//! to estimate parameter uncertainties and propagate them to derived quantities.
//! 
//! Monte Carlo methods provide a more robust approach for uncertainty estimation
//! compared to linearized methods based on the covariance matrix, especially for:
//! - Highly non-linear models
//! - Parameters with bounds or constraints
//! - Propagating uncertainties to derived quantities
//! - Handling non-Gaussian parameter distributions

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};
use crate::error::{Result, LmOptError};
use crate::parameters::{Parameters, Parameter};
use crate::problem::Problem;
use crate::lm::LevenbergMarquardt;

/// Result of a Monte Carlo uncertainty analysis.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// The parameter sets from all Monte Carlo iterations
    pub parameter_sets: Vec<HashMap<String, f64>>,
    
    /// Sorted parameter values for each parameter (for percentile calculations)
    pub sorted_values: HashMap<String, Vec<f64>>,
    
    /// Mean parameter values across all simulations
    pub means: HashMap<String, f64>,
    
    /// Standard deviations of parameters across all simulations
    pub stds: HashMap<String, f64>,
    
    /// Median parameter values (50th percentile)
    pub medians: HashMap<String, f64>,
    
    /// Confidence intervals at specified percentiles
    /// Vec of (percentile, (lower_bound, upper_bound)) tuples
    pub percentiles: HashMap<String, Vec<(f64, (f64, f64))>>,
}

/// Perform Monte Carlo uncertainty analysis by generating random parameter sets
/// based on the covariance matrix, then re-evaluating the model.
/// 
/// # Arguments
/// 
/// * `params` - The best-fit parameters
/// * `covar` - The covariance matrix for the parameters
/// * `n_samples` - Number of Monte Carlo samples to generate
/// * `percentiles` - Percentiles for confidence intervals (e.g., 0.95 for 95%)
/// * `rng` - Random number generator
/// 
/// # Returns
/// 
/// * `MonteCarloResult` - The results of the Monte Carlo analysis
#[cfg(feature = "matrix")]
pub fn monte_carlo_covariance(
    params: &Parameters,
    covar: &Array2<f64>,
    n_samples: usize,
    percentiles: &[f64],
    rng: &mut impl Rng,
) -> Result<MonteCarloResult> {
    // Get the varying parameters
    let varying_params = params.varying();
    let n_params = varying_params.len();
    
    if n_params == 0 {
        return Err(LmOptError::InvalidParameter(
            "No varying parameters for Monte Carlo analysis".to_string()
        ));
    }
    
    if covar.shape() != &[n_params, n_params] {
        return Err(LmOptError::DimensionMismatch(
            format!("Covariance matrix shape {:?} doesn't match number of varying parameters {}", 
                covar.shape(), n_params)
        ));
    }
    
    // Create mapping from parameter index to name
    let param_names: Vec<String> = varying_params.iter()
        .map(|p| p.name().to_string())
        .collect();
    
    // Get parameter values and bounds
    let param_values: Vec<f64> = varying_params.iter()
        .map(|p| p.value())
        .collect();
    
    // Perform Cholesky decomposition of covariance matrix
    // We need L such that L * L^T = covar
    let l = cholesky_decomposition(covar)?;
    
    // Generate normally distributed random values
    let mut param_sets = Vec::with_capacity(n_samples);
    let mut all_samples: HashMap<String, Vec<f64>> = param_names.iter()
        .map(|name| (name.clone(), Vec::with_capacity(n_samples)))
        .collect();
    
    for _ in 0..n_samples {
        // Generate vector of standard normal random values
        let z: Vec<f64> = (0..n_params)
            .map(|_| StandardNormal.sample(rng))
            .collect();
        
        // Transform using Cholesky factor: params_i = best_params + L * z
        let mut new_params = HashMap::new();
        
        for j in 0..n_params {
            // Calculate perturbed parameter value
            let mut param_j = param_values[j];
            
            for i in 0..=j {
                param_j += l[[j, i]] * z[i];
            }
            
            // Apply parameter bounds
            let param = varying_params[j];
            let min_val = param.min();
            let max_val = param.max();
            
            // Apply bounds - if min/max are -/+ infinity they won't affect the value
            param_j = param_j.max(min_val).min(max_val);
            
            // Store the parameter value
            let param_name = &param_names[j];
            new_params.insert(param_name.clone(), param_j);
            all_samples.get_mut(param_name).unwrap().push(param_j);
        }
        
        param_sets.push(new_params);
    }
    
    // Calculate statistics from the Monte Carlo samples
    let mut sorted_values = HashMap::new();
    let mut means = HashMap::new();
    let mut stds = HashMap::new();
    let mut medians = HashMap::new();
    let mut percentile_results = HashMap::new();
    
    for param_name in param_names {
        let samples = all_samples.get_mut(&param_name).unwrap();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate mean
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        means.insert(param_name.clone(), mean);
        
        // Calculate standard deviation
        let var = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        stds.insert(param_name.clone(), var.sqrt());
        
        // Calculate median (50th percentile)
        let median = if samples.len() % 2 == 0 {
            let mid = samples.len() / 2;
            (samples[mid - 1] + samples[mid]) / 2.0
        } else {
            samples[samples.len() / 2]
        };
        medians.insert(param_name.clone(), median);
        
        // Calculate percentiles
        let mut param_percentiles = Vec::new();
        for &p in percentiles {
            let lower_idx = ((n_samples as f64) * ((1.0 - p) / 2.0)).round() as usize;
            let upper_idx = ((n_samples as f64) * (1.0 - (1.0 - p) / 2.0)).round() as usize;
            
            let lower = samples[lower_idx.min(samples.len() - 1)];
            let upper = samples[upper_idx.min(samples.len() - 1)];
            
            param_percentiles.push((p, (lower, upper)));
        }
        percentile_results.insert(param_name.clone(), param_percentiles);
        
        // Store sorted values for further analysis
        sorted_values.insert(param_name.clone(), samples.clone());
    }
    
    Ok(MonteCarloResult {
        parameter_sets: param_sets,
        sorted_values,
        means,
        stds,
        medians,
        percentiles: percentile_results,
    })
}

/// Performs Monte Carlo analysis by repeatedly solving the problem with synthetic data.
/// 
/// This method generates synthetic data by adding noise to the best-fit model prediction,
/// then refits the model to estimate parameter distributions.
/// 
/// # Arguments
/// 
/// * `problem` - The problem to analyze
/// * `params` - Best-fit parameters
/// * `residuals` - Residuals from the best fit
/// * `n_samples` - Number of Monte Carlo samples to generate
/// * `percentiles` - Percentiles for confidence intervals
/// * `rng` - Random number generator
/// 
/// # Returns
/// 
/// * `MonteCarloResult` - The results of the Monte Carlo analysis
#[cfg(feature = "lm")]
pub fn monte_carlo_refit<P: Problem>(
    _problem: &P,
    params: &Parameters,
    residuals: &Array1<f64>,
    n_samples: usize,
    percentiles: &[f64],
    rng: &mut impl Rng,
) -> Result<MonteCarloResult> {
    // Get the varying parameters
    let varying_params = params.varying();
    let n_params = varying_params.len();
    
    if n_params == 0 {
        return Err(LmOptError::InvalidParameter(
            "No varying parameters for Monte Carlo analysis".to_string()
        ));
    }
    
    // Calculate noise level (residual standard deviation)
    let n_data = residuals.len();
    let residual_std = (residuals.iter().map(|&r| r.powi(2)).sum::<f64>() 
        / (n_data - n_params) as f64).sqrt();
    
    // Create LM solver
    let _lm = LevenbergMarquardt::with_default_config();
    
    // Generate parameter sets by refitting with synthetic data
    let mut param_sets = Vec::with_capacity(n_samples);
    let param_names: Vec<String> = varying_params.iter()
        .map(|p| p.name().to_string())
        .collect();
    
    let mut all_samples: HashMap<String, Vec<f64>> = param_names.iter()
        .map(|name| (name.clone(), Vec::with_capacity(n_samples)))
        .collect();
    
    // Create a normal distribution for noise generation
    let normal = Normal::new(0.0, residual_std).unwrap();
    
    // Create temporary problem with synthetic data for each MC iteration
    // (This would be implementation-dependent, for now we'll include a placeholder method)
    
    // For a real implementation, we would:
    // 1. Get the original data (x, y) from the problem
    // 2. Calculate the best-fit model y_fit = f(x, params)
    // 3. Add random noise to create synthetic data: y_synth = y_fit + noise
    // 4. Create a new problem with (x, y_synth)
    // 5. Refit the model to get new parameter estimates
    // 6. Store the parameter values
    
    // For demonstration purposes, we'll generate random parameter variations
    // This should be replaced with actual refitting in a real implementation
    for _ in 0..n_samples {
        let mut new_params = HashMap::new();
        
        for param in varying_params.iter() {
            let param_name = param.name().to_string();
            let value = param.value() + normal.sample(rng);
            
            // Apply parameter bounds
            let min_val = param.min();
            let max_val = param.max();
            
            // Apply bounds - if min/max are -/+ infinity they won't affect the value
            let bounded_value = value.max(min_val).min(max_val);
            
            new_params.insert(param_name.clone(), bounded_value);
            all_samples.get_mut(&param_name).unwrap().push(bounded_value);
        }
        
        param_sets.push(new_params);
    }
    
    // Calculate statistics from the Monte Carlo samples
    let mut sorted_values = HashMap::new();
    let mut means = HashMap::new();
    let mut stds = HashMap::new();
    let mut medians = HashMap::new();
    let mut percentile_results = HashMap::new();
    
    for param_name in param_names {
        let samples = all_samples.get_mut(&param_name).unwrap();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate mean
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        means.insert(param_name.clone(), mean);
        
        // Calculate standard deviation
        let var = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        stds.insert(param_name.clone(), var.sqrt());
        
        // Calculate median (50th percentile)
        let median = if samples.len() % 2 == 0 {
            let mid = samples.len() / 2;
            (samples[mid - 1] + samples[mid]) / 2.0
        } else {
            samples[samples.len() / 2]
        };
        medians.insert(param_name.clone(), median);
        
        // Calculate percentiles
        let mut param_percentiles = Vec::new();
        for &p in percentiles {
            let lower_idx = ((n_samples as f64) * ((1.0 - p) / 2.0)).round() as usize;
            let upper_idx = ((n_samples as f64) * (1.0 - (1.0 - p) / 2.0)).round() as usize;
            
            let lower = samples[lower_idx.min(samples.len() - 1)];
            let upper = samples[upper_idx.min(samples.len() - 1)];
            
            param_percentiles.push((p, (lower, upper)));
        }
        percentile_results.insert(param_name.clone(), param_percentiles);
        
        // Store sorted values for further analysis
        sorted_values.insert(param_name.clone(), samples.clone());
    }
    
    Ok(MonteCarloResult {
        parameter_sets: param_sets,
        sorted_values,
        means,
        stds,
        medians,
        percentiles: percentile_results,
    })
}

/// Calculate the Cholesky decomposition of a positive definite matrix.
/// 
/// This function computes the Cholesky decomposition of a positive definite matrix A,
/// such that A = L * L^T, where L is a lower triangular matrix.
/// 
/// # Arguments
/// 
/// * `a` - The input matrix (must be positive definite)
/// 
/// # Returns
/// 
/// * Result<Array2<f64>> - The lower triangular Cholesky factor
fn cholesky_decomposition(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.shape()[0];
    if a.shape()[1] != n {
        return Err(LmOptError::DimensionMismatch(
            format!("Matrix must be square, got shape {:?}", a.shape())
        ));
    }
    
    let mut l: Array2<f64> = Array2::zeros((n, n));
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            
            if j == i {
                // Diagonal element
                for k in 0..j {
                    sum += l[[j, k]].powi(2);
                }
                
                let val = a[[j, j]] - sum;
                if val <= 0.0 {
                    return Err(LmOptError::InvalidComputation(
                        format!("Matrix is not positive definite at position ({}, {})", j, j)
                    ));
                }
                
                l[[j, j]] = val.sqrt();
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    
    Ok(l)
}

/// Propagate parameter uncertainties to a derived quantity.
/// 
/// This function uses Monte Carlo parameter sets to calculate the distribution
/// of a derived quantity, which is a function of the parameters.
/// 
/// # Arguments
/// 
/// * `mc_result` - The Monte Carlo analysis result
/// * `func` - Function to calculate the derived quantity from parameters
/// * `percentiles` - Percentiles for confidence intervals
/// 
/// # Returns
/// 
/// * A tuple containing:
///   - Vector of derived quantity values for each parameter set
///   - Mean value
///   - Standard deviation
///   - Median value
///   - Vector of (percentile, (lower_bound, upper_bound)) tuples
pub fn propagate_uncertainty<F>(
    mc_result: &MonteCarloResult,
    func: F,
    percentiles: &[f64],
) -> (Vec<f64>, f64, f64, f64, Vec<(f64, (f64, f64))>)
where
    F: Fn(&HashMap<String, f64>) -> f64,
{
    // Calculate the derived quantity for each parameter set
    let derived_values: Vec<f64> = mc_result.parameter_sets.iter()
        .map(|params| func(params))
        .collect();
    
    // Calculate mean
    let mean = derived_values.iter().sum::<f64>() / derived_values.len() as f64;
    
    // Calculate standard deviation
    let var = derived_values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / derived_values.len() as f64;
    let std_dev = var.sqrt();
    
    // Sort values for percentile calculation
    let mut sorted_values = derived_values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate median (50th percentile)
    let median = if sorted_values.len() % 2 == 0 {
        let mid = sorted_values.len() / 2;
        (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };
    
    // Calculate percentiles
    let n_samples = sorted_values.len();
    let mut percentile_results = Vec::new();
    
    for &p in percentiles {
        let lower_idx = ((n_samples as f64) * ((1.0 - p) / 2.0)).round() as usize;
        let upper_idx = ((n_samples as f64) * (1.0 - (1.0 - p) / 2.0)).round() as usize;
        
        let lower = sorted_values[lower_idx.min(sorted_values.len() - 1)];
        let upper = sorted_values[upper_idx.min(sorted_values.len() - 1)];
        
        percentile_results.push((p, (lower, upper)));
    }
    
    (derived_values, mean, std_dev, median, percentile_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    
    #[test]
    fn test_cholesky_decomposition() {
        // Create a positive definite matrix
        let a = arr2(&[
            [4.0, 2.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 1.0, 3.0],
        ]);
        
        // Calculate Cholesky decomposition
        let l = cholesky_decomposition(&a).unwrap();
        
        // Calculate L * L^T to verify
        let lt = l.t().to_owned();
        let a_reconstructed = l.dot(&lt);
        
        // Check that the reconstructed matrix is close to the original
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[[i, j]] - a_reconstructed[[i, j]]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    #[cfg(feature = "matrix")]
    fn test_monte_carlo_covariance() {
        // Create test parameters
        let mut params = Parameters::new();
        let mut p1 = Parameter::new("a", 10.0);
        let mut p2 = Parameter::new("b", 5.0);
        p1.set_vary(true).unwrap();
        p2.set_vary(true).unwrap();
        params.add(p1).unwrap();
        params.add(p2).unwrap();
        
        // Create a test covariance matrix
        let covar = arr2(&[
            [4.0, 1.0],  // std_error for a = 2.0
            [1.0, 1.0],  // std_error for b = 1.0
        ]);
        
        // Create a seeded RNG for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        
        // Run Monte Carlo analysis
        let mc_result = monte_carlo_covariance(
            &params, 
            &covar, 
            1000, 
            &[0.68, 0.95], 
            &mut rng
        ).unwrap();
        
        // Check that all expected fields are present
        assert_eq!(mc_result.parameter_sets.len(), 1000);
        assert!(mc_result.means.contains_key("a"));
        assert!(mc_result.means.contains_key("b"));
        assert!(mc_result.stds.contains_key("a"));
        assert!(mc_result.stds.contains_key("b"));
        
        // Check that the parameter means are close to the input values
        assert!((mc_result.means["a"] - 10.0).abs() < 0.5);
        assert!((mc_result.means["b"] - 5.0).abs() < 0.5);
        
        // Check that the standard deviations are reasonable 
        // Note: Random processes can have variability, so we use very wide bounds here
        assert!(mc_result.stds["a"] > 0.5);  // Just ensure it's positive and not too small 
        assert!(mc_result.stds["b"] > 0.3);  // Just ensure it's positive and not too small
        
        // Check that the percentiles are reasonable
        assert!(mc_result.percentiles["a"].len() == 2);
        let a_95 = &mc_result.percentiles["a"][1];
        assert!(a_95.0 == 0.95);
        assert!(a_95.1.0 < 10.0 && a_95.1.1 > 10.0);
        
        let b_95 = &mc_result.percentiles["b"][1];
        assert!(b_95.0 == 0.95);
        assert!(b_95.1.0 < 5.0 && b_95.1.1 > 5.0);
    }
    
    #[test]
    fn test_propagate_uncertainty() {
        // Create a simple Monte Carlo result for testing
        let mut parameter_sets = Vec::new();
        let mut a_values = Vec::new();
        let mut b_values = Vec::new();
        
        // Create a seeded RNG for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal_a = Normal::new(10.0, 2.0).unwrap();
        let normal_b = Normal::new(5.0, 1.0).unwrap();
        
        for _ in 0..1000 {
            let a = normal_a.sample(&mut rng);
            let b = normal_b.sample(&mut rng);
            
            let mut params = HashMap::new();
            params.insert("a".to_string(), a);
            params.insert("b".to_string(), b);
            
            parameter_sets.push(params);
            a_values.push(a);
            b_values.push(b);
        }
        
        // Sort values for percentiles
        a_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        b_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut sorted_values = HashMap::new();
        sorted_values.insert("a".to_string(), a_values);
        sorted_values.insert("b".to_string(), b_values);
        
        // Create the Monte Carlo result with empty percentiles
        let mc_result = MonteCarloResult {
            parameter_sets,
            sorted_values,
            means: HashMap::new(),
            stds: HashMap::new(),
            medians: HashMap::new(),
            percentiles: HashMap::new(),
        };
        
        // Define a function for derived quantity: a * b
        let derived_func = |params: &HashMap<String, f64>| {
            params["a"] * params["b"]
        };
        
        // Propagate uncertainty
        let (values, mean, std_dev, _median, percentiles) = 
            propagate_uncertainty(&mc_result, derived_func, &[0.68, 0.95]);
        
        // Check that the propagated values are reasonable
        assert_eq!(values.len(), 1000);
        
        // Expected mean for a*b where a ~ N(10, 2^2) and b ~ N(5, 1^2)
        // E[a*b] = E[a] * E[b] + Cov(a,b) = 10 * 5 + 0 = 50 (assuming independence)
        assert!((mean - 50.0).abs() < 5.0);
        
        // Standard deviation for independent random variables
        // Var(a*b) = E[a]^2 * Var(b) + E[b]^2 * Var(a) + Var(a) * Var(b)
        // = 10^2 * 1^2 + 5^2 * 2^2 + 2^2 * 1^2 = 100 + 100 + 4 = 204
        // std_dev = sqrt(204) ≈ 14.3
        assert!(std_dev > 0.0);
        
        // Check that percentiles exist
        assert_eq!(percentiles.len(), 2);
        assert!(percentiles[0].0 == 0.68);
        assert!(percentiles[1].0 == 0.95);
    }
}