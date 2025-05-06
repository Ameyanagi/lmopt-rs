//! # Uncertainty Calculation
//! 
//! This module provides functionality for calculating uncertainties in parameter
//! estimates from nonlinear least-squares optimization results. It includes:
//! 
//! - Covariance matrix estimation from Jacobian matrices
//! - Confidence interval calculation for parameters
//! - Standard error calculation for parameter estimates
//! - F-test for comparing fits with different parameters fixed
//! 
//! The implementation approach is similar to that of lmfit-py, providing
//! robust uncertainty quantification tools for fitted parameters.

mod covariance;
mod confidence;
mod monte_carlo;

pub use covariance::{
    calculate_covariance,
    calculate_correlation,
    standard_errors_from_covariance,
};

pub use confidence::{
    confidence_intervals,
    confidence_regions_2d,
    sigma_to_probability,
    probability_to_sigma,
};

pub use monte_carlo::{
    monte_carlo_covariance,
    monte_carlo_refit,
    propagate_uncertainty,
    MonteCarloResult,
};

use ndarray::{Array1, Array2};
use crate::error::Result;
use crate::parameters::Parameters;
use std::collections::HashMap;

/// Represents a confidence interval value.
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// The probability level (e.g., 0.68 for 1-sigma)
    pub probability: f64,
    /// The lower bound of the confidence interval
    pub lower: f64,
    /// The upper bound of the confidence interval
    pub upper: f64,
}

/// Structure to hold uncertainty calculation results.
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Covariance matrix for the parameters
    pub covariance: Array2<f64>,
    /// Standard errors for each parameter
    pub standard_errors: HashMap<String, f64>,
    /// Confidence intervals at different probability levels
    pub confidence_intervals: HashMap<String, Vec<ConfidenceInterval>>,
    /// Correlation matrix derived from covariance
    pub correlation: Array2<f64>,
    /// Chi-square value at minimum
    pub chisqr: f64,
    /// Reduced chi-square (chi^2 / nfree)
    pub redchi: f64,
    /// Degrees of freedom (n_points - n_parameters)
    pub nfree: usize,
    /// Monte Carlo analysis results (if performed)
    pub monte_carlo: Option<monte_carlo::MonteCarloResult>,
}

/// Calculator for parameter uncertainties.
#[derive(Debug, Clone)]
pub struct UncertaintyCalculator {
    /// Degrees of freedom (n_points - n_parameters)
    pub nfree: usize,
    /// Chi-square value at minimum
    pub chisqr: f64,
    /// Reduced chi-square (chi^2 / nfree)
    pub redchi: f64,
}

impl UncertaintyCalculator {
    /// Create a new UncertaintyCalculator
    pub fn new(ndata: usize, nvarys: usize, chisqr: f64) -> Self {
        let nfree = if ndata > nvarys { ndata - nvarys } else { 1 };
        let redchi = chisqr / nfree as f64;
        
        Self {
            nfree,
            chisqr,
            redchi,
        }
    }
    
    /// Calculate the covariance matrix from the Jacobian
    #[cfg(feature = "matrix")]
    pub fn calculate_covariance(&self, jacobian: &Array2<f64>) -> Result<Array2<f64>> {
        covariance::calculate_covariance(jacobian, self.redchi)
    }
    
    /// Calculate standard errors from covariance matrix
    pub fn calculate_standard_errors(&self, covar: &Array2<f64>, params: &Parameters) -> HashMap<String, f64> {
        let std_errors = covariance::standard_errors_from_covariance(covar);
        let mut errors = HashMap::new();
        
        // Get only the varying parameters
        let varying_params = params.varying();
        
        for (i, param) in varying_params.iter().enumerate() {
            if i < std_errors.len() {
                errors.insert(param.name().to_string(), std_errors[i]);
            }
        }
        
        errors
    }
    
    /// Calculate correlation matrix from covariance matrix
    pub fn calculate_correlation(&self, covar: &Array2<f64>) -> Array2<f64> {
        covariance::calculate_correlation(covar)
    }

    /// Calculate confidence intervals for parameters
    pub fn calculate_confidence_intervals(
        &self,
        params: &Parameters,
        covar: &Array2<f64>,
        sigmas: &[f64],
    ) -> Result<HashMap<String, Vec<ConfidenceInterval>>> {
        confidence::confidence_intervals(params, covar, sigmas, None)
    }
    
    /// Calculate 2D confidence regions for a pair of parameters
    pub fn calculate_confidence_regions_2d(
        &self,
        params: &Parameters,
        param1_name: &str,
        param2_name: &str,
        nx: usize,
        ny: usize,
        nsigma: f64,
        covar: &Array2<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, Array2<f64>)> {
        confidence::confidence_regions_2d(params, param1_name, param2_name, nx, ny, nsigma, covar)
    }
    
    /// Perform Monte Carlo uncertainty analysis using the covariance matrix
    #[cfg(feature = "matrix")]
    pub fn monte_carlo_analysis(
        &self,
        params: &Parameters,
        covar: &Array2<f64>,
        n_samples: usize,
        percentiles: &[f64],
        rng: &mut impl rand::Rng,
    ) -> Result<monte_carlo::MonteCarloResult> {
        monte_carlo::monte_carlo_covariance(params, covar, n_samples, percentiles, rng)
    }
    
    /// Perform Monte Carlo uncertainty analysis by refitting with synthetic data
    #[cfg(feature = "lm")]
    pub fn monte_carlo_refit_analysis<P: crate::problem::Problem>(
        &self,
        problem: &P,
        params: &Parameters,
        residuals: &Array1<f64>,
        n_samples: usize,
        percentiles: &[f64],
        rng: &mut impl rand::Rng,
    ) -> Result<monte_carlo::MonteCarloResult> {
        monte_carlo::monte_carlo_refit(problem, params, residuals, n_samples, percentiles, rng)
    }
    
    /// F-test for comparing nested models
    pub fn f_test(&self, chisqr_1: f64, nfree_1: usize, chisqr_2: f64, nfree_2: usize) -> f64 {
        if nfree_1 <= nfree_2 {
            return 1.0; // Cannot compute F-test when model 1 has fewer/equal free parameters
        }
        
        let nfix = nfree_1 - nfree_2; // Number of parameters fixed
        let dchi = chisqr_2 / chisqr_1 - 1.0;
        
        // Calculate F-statistic
        let f_stat = dchi * (nfree_2 as f64) / (nfix as f64);
        
        // Calculate probability from F-distribution (not implemented yet)
        // This will be a crate dependency like statrs for the F-distribution CDF
        // For now, return the F-statistic
        f_stat
    }
}

/// Calculate covariance matrix from Jacobian matrix
#[cfg(feature = "matrix")]
pub fn covariance_matrix(jacobian: &Array2<f64>, chisqr: f64, ndata: usize, nvarys: usize) -> Result<Array2<f64>> {
    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);
    calculator.calculate_covariance(jacobian)
}

/// Calculate standard errors from covariance matrix
pub fn standard_errors(
    covar: &Array2<f64>, 
    params: &Parameters
) -> HashMap<String, f64> {
    // Get standard errors from the covariance matrix
    let std_errors = covariance::standard_errors_from_covariance(covar);
    let mut errors = HashMap::new();
    
    // Get only the varying parameters
    let varying_params = params.varying();
    
    // Map standard errors to parameter names
    for (i, param) in varying_params.iter().enumerate() {
        if i < std_errors.len() {
            errors.insert(param.name().to_string(), std_errors[i]);
        }
    }
    
    errors
}

/// Create a complete uncertainty analysis
#[cfg(feature = "matrix")]
pub fn uncertainty_analysis(
    jacobian: &Array2<f64>,
    params: &Parameters,
    chisqr: f64,
    ndata: usize,
    sigmas: &[f64],
) -> Result<UncertaintyResult> {
    let nvarys = params.varying().len();
    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);
    
    let covar = calculator.calculate_covariance(jacobian)?;
    let std_errors = calculator.calculate_standard_errors(&covar, params);
    let correlation = calculator.calculate_correlation(&covar);
    let confidence_intervals = calculator.calculate_confidence_intervals(params, &covar, sigmas)?;
    
    Ok(UncertaintyResult {
        covariance: covar,
        standard_errors: std_errors,
        confidence_intervals,
        correlation,
        chisqr,
        redchi: calculator.redchi,
        nfree: calculator.nfree,
        monte_carlo: None,
    })
}

/// Create a complete uncertainty analysis with Monte Carlo simulation
#[cfg(feature = "matrix")]
pub fn uncertainty_analysis_with_monte_carlo(
    jacobian: &Array2<f64>,
    params: &Parameters,
    chisqr: f64,
    ndata: usize,
    sigmas: &[f64],
    n_samples: usize,
    percentiles: &[f64],
    rng: &mut impl rand::Rng,
) -> Result<UncertaintyResult> {
    let nvarys = params.varying().len();
    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);
    
    let covar = calculator.calculate_covariance(jacobian)?;
    let std_errors = calculator.calculate_standard_errors(&covar, params);
    let correlation = calculator.calculate_correlation(&covar);
    let confidence_intervals = calculator.calculate_confidence_intervals(params, &covar, sigmas)?;
    
    // Perform Monte Carlo analysis
    let mc_result = calculator.monte_carlo_analysis(params, &covar, n_samples, percentiles, rng)?;
    
    Ok(UncertaintyResult {
        covariance: covar,
        standard_errors: std_errors,
        confidence_intervals,
        correlation,
        chisqr,
        redchi: calculator.redchi,
        nfree: calculator.nfree,
        monte_carlo: Some(mc_result),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use crate::parameters::{Parameter, Parameters};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    
    #[test]
    #[cfg(feature = "matrix")]
    fn test_covariance_calculation() {
        // Create a simple Jacobian (2 parameters, 3 data points)
        let jacobian = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        
        let chisqr = 2.0;
        let ndata = 3;
        let nvarys = 2;
        
        let covar = covariance_matrix(&jacobian, chisqr, ndata, nvarys).unwrap();
        
        // Verify dimensions
        assert_eq!(covar.shape(), &[2, 2]);
        
        // Values should be scaled by reduced chi-square = 2.0 / (3 - 2) = 2.0
        let _expected_gram = arr2(&[
            [35.0, 47.0],
            [47.0, 64.0],
        ]);
        
        // Expected covariance is approximately 2.0 * inv(J^T * J)
        // Note: This test only checks shape and rough scaling, not exact values
        assert!(covar[[0, 0]] > 0.0);
        assert!(covar[[1, 1]] > 0.0);
        assert_eq!(covar[[0, 1]], covar[[1, 0]]);
    }
    
    #[test]
    #[cfg(feature = "matrix")]
    fn test_uncertainty_analysis() {
        // Create a simple Jacobian (2 parameters, 3 data points)
        let jacobian = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        
        // Create test parameters
        let mut params = Parameters::new();
        let mut p1 = Parameter::new("a", 10.0);
        let mut p2 = Parameter::new("b", 5.0);
        p1.set_vary(true).unwrap();
        p2.set_vary(true).unwrap();
        params.add(p1).unwrap();
        params.add(p2).unwrap();
        
        let chisqr = 2.0;
        let ndata = 3;
        let sigmas = &[1.0, 2.0];
        
        // Run full uncertainty analysis
        let result = uncertainty_analysis(&jacobian, &params, chisqr, ndata, sigmas).unwrap();
        
        // Check that all components are present
        assert_eq!(result.covariance.shape(), &[2, 2]);
        assert_eq!(result.correlation.shape(), &[2, 2]);
        assert_eq!(result.standard_errors.len(), 2);
        assert_eq!(result.confidence_intervals.len(), 2);
        assert_eq!(result.chisqr, 2.0);
        assert_eq!(result.nfree, 1); // ndata (3) - nvarys (2) = 1
        assert_eq!(result.redchi, 2.0); // chisqr (2.0) / nfree (1) = 2.0
        assert!(result.monte_carlo.is_none()); // No Monte Carlo analysis performed
        
        // Check that standard errors were calculated
        assert!(result.standard_errors.contains_key("a"));
        assert!(result.standard_errors.contains_key("b"));
        
        // Check that confidence intervals were calculated
        assert!(result.confidence_intervals.contains_key("a"));
        assert!(result.confidence_intervals.contains_key("b"));
        assert_eq!(result.confidence_intervals["a"].len(), 2); // Two sigma levels
        assert_eq!(result.confidence_intervals["b"].len(), 2);
    }
    
    #[test]
    #[cfg(feature = "matrix")]
    fn test_uncertainty_analysis_with_monte_carlo() {
        // Create a simple Jacobian (2 parameters, 3 data points)
        let jacobian = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        
        // Create test parameters
        let mut params = Parameters::new();
        let mut p1 = Parameter::new("a", 10.0);
        let mut p2 = Parameter::new("b", 5.0);
        p1.set_vary(true).unwrap();
        p2.set_vary(true).unwrap();
        params.add(p1).unwrap();
        params.add(p2).unwrap();
        
        let chisqr = 2.0;
        let ndata = 3;
        let sigmas = &[1.0, 2.0];
        let n_samples = 100; // Small number for test speed
        let percentiles = &[0.68, 0.95];
        
        // Create a seeded RNG for reproducibility
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        
        // Run full uncertainty analysis with Monte Carlo
        let result = uncertainty_analysis_with_monte_carlo(
            &jacobian, &params, chisqr, ndata, sigmas, n_samples, percentiles, &mut rng
        ).unwrap();
        
        // Check that all components are present
        assert_eq!(result.covariance.shape(), &[2, 2]);
        assert_eq!(result.correlation.shape(), &[2, 2]);
        assert_eq!(result.standard_errors.len(), 2);
        assert_eq!(result.confidence_intervals.len(), 2);
        assert_eq!(result.chisqr, 2.0);
        assert_eq!(result.nfree, 1); // ndata (3) - nvarys (2) = 1
        assert_eq!(result.redchi, 2.0); // chisqr (2.0) / nfree (1) = 2.0
        
        // Check that Monte Carlo results are present
        assert!(result.monte_carlo.is_some());
        let mc_result = result.monte_carlo.unwrap();
        assert_eq!(mc_result.parameter_sets.len(), n_samples);
        assert!(mc_result.means.contains_key("a"));
        assert!(mc_result.means.contains_key("b"));
        
        // Check that the percentiles are reasonable
        assert!(mc_result.percentiles.contains_key("a"));
        assert!(mc_result.percentiles.contains_key("b"));
        assert_eq!(mc_result.percentiles["a"].len(), 2);
        assert_eq!(mc_result.percentiles["a"][0].0, 0.68);
        assert_eq!(mc_result.percentiles["a"][1].0, 0.95);
    }
    
    #[test]
    fn test_standard_errors() {
        // Create a test covariance matrix
        let covar = arr2(&[
            [0.2, 0.05],  // Using 0.2 to match the expected sqrt(0.2) = 0.4472
            [0.05, 0.2],
        ]);
        
        // Create some parameters
        let mut params = Parameters::new();
        let p1 = Parameter::new("p1", 1.0);
        let p2 = Parameter::new("p2", 2.0);
        // Set parameters to vary
        let mut p1_with_vary = p1;
        p1_with_vary.set_vary(true).unwrap();
        let mut p2_with_vary = p2;
        p2_with_vary.set_vary(true).unwrap();
        
        params.add(p1_with_vary).unwrap();
        params.add(p2_with_vary).unwrap();
        
        let errors = super::standard_errors(&covar, &params);
        
        // Check that standard errors were calculated correctly
        assert!(errors.contains_key("p1"));
        assert!(errors.contains_key("p2"));
        
        // Check that the error values are approximately the square root of the covariance diagonal
        let p1_expected = 0.2f64.sqrt();  // ~0.447
        let p2_expected = 0.2f64.sqrt();  // ~0.447
        let tolerance = 1e-10;
        
        assert!((errors["p1"] - p1_expected).abs() < tolerance, 
                "p1 error should be {}, got {}", p1_expected, errors["p1"]);
        assert!((errors["p2"] - p2_expected).abs() < tolerance,
                "p2 error should be {}, got {}", p2_expected, errors["p2"]);
    }
    
    #[test]
    fn test_correlation_matrix() {
        // Create a test covariance matrix
        let covar = arr2(&[
            [0.1, 0.05],
            [0.05, 0.2],
        ]);
        
        let calculator = UncertaintyCalculator::new(0, 0, 0.0);
        let correl = calculator.calculate_correlation(&covar);
        
        // Check correlation matrix properties
        assert_eq!(correl.shape(), &[2, 2]);
        assert_eq!(correl[[0, 0]], 1.0);
        assert_eq!(correl[[1, 1]], 1.0);
        
        // Off-diagonal elements should be covar_ij / sqrt(covar_ii * covar_jj)
        let expected = 0.05 / (0.1f64 * 0.2f64).sqrt();
        assert!((correl[[0, 1]] - expected).abs() < 1e-10);
        assert!((correl[[1, 0]] - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_f_test() {
        let calculator = UncertaintyCalculator::new(0, 0, 0.0);
        
        // Model 1: chisqr = 100, nfree = 10
        // Model 2: chisqr = 110, nfree = 9 (one parameter fixed)
        let f_stat = calculator.f_test(100.0, 10, 110.0, 9);
        
        // f = (chisqr_2/chisqr_1 - 1) * nfree_2 / (nfree_1 - nfree_2)
        // f = (110/100 - 1) * 9 / (10 - 9) = (1.1 - 1) * 9 / 1 = 0.1 * 9 = 0.9
        assert!((f_stat - 0.9).abs() < 1e-10);
    }
}