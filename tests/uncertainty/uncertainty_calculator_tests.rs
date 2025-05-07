//! Tests for the UncertaintyCalculator struct
//!
//! This file contains tests for the UncertaintyCalculator struct,
//! which provides a high-level interface for calculating parameter
//! uncertainties in nonlinear least-squares fits.

use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::uncertainty::{
    uncertainty_analysis, uncertainty_analysis_with_monte_carlo, UncertaintyCalculator,
    UncertaintyResult,
};
use ndarray::{arr1, arr2, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// Helper function to create test parameters and Jacobian for a quadratic model
fn create_test_data() -> (Parameters, Array2<f64>, f64, usize) {
    // Create test parameters for y = a*x^2 + b*x + c
    let mut params = Parameters::new();
    let mut a = Parameter::new("a", 2.0);
    let mut b = Parameter::new("b", 5.0);
    let mut c = Parameter::new("c", 1.0);

    a.set_vary(true).unwrap();
    b.set_vary(true).unwrap();
    c.set_vary(true).unwrap();

    params.add(a).unwrap();
    params.add(b).unwrap();
    params.add(c).unwrap();

    // Create Jacobian for x values [-2, -1, 0, 1, 2]
    // Each row is a data point, each column is a parameter derivative
    // J = [∂r/∂a, ∂r/∂b, ∂r/∂c] = [-x^2, -x, -1]
    let jacobian = arr2(&[
        [4.0, 2.0, 1.0],  // x = -2: [-(-2)^2, -(-2), -1]
        [1.0, 1.0, 1.0],  // x = -1: [-(-1)^2, -(-1), -1]
        [0.0, 0.0, 1.0],  // x = 0: [-0^2, -0, -1]
        [1.0, -1.0, 1.0], // x = 1: [-1^2, -1, -1]
        [4.0, -2.0, 1.0], // x = 2: [-2^2, -2, -1]
    ]);

    // Other parameters
    let chisqr = 2.5; // Arbitrary chi-square value
    let ndata = 5; // 5 data points

    (params, jacobian, chisqr, ndata)
}

#[test]
#[cfg(feature = "matrix")]
fn test_uncertainty_calculator_new() {
    let (_, _, chisqr, ndata) = create_test_data();
    let nvarys = 3; // Three varying parameters

    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);

    // Check that the calculator was initialized correctly
    assert_eq!(calculator.nfree, ndata - nvarys); // 5 - 3 = 2
    assert_eq!(calculator.chisqr, chisqr);
    assert_eq!(calculator.redchi, chisqr / (ndata - nvarys) as f64); // 2.5 / 2 = 1.25
}

#[test]
#[cfg(feature = "matrix")]
fn test_uncertainty_calculator_methods() {
    let (params, jacobian, chisqr, ndata) = create_test_data();
    let nvarys = params.varying().len();

    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);

    // Test covariance calculation
    let covar = calculator.calculate_covariance(&jacobian).unwrap();
    assert_eq!(covar.shape(), &[nvarys, nvarys]);

    // Test standard errors calculation
    let std_errors = calculator.calculate_standard_errors(&covar, &params);
    assert_eq!(std_errors.len(), nvarys);
    assert!(std_errors.contains_key("a"));
    assert!(std_errors.contains_key("b"));
    assert!(std_errors.contains_key("c"));

    // Test correlation calculation
    let correlation = calculator.calculate_correlation(&covar);
    assert_eq!(correlation.shape(), &[nvarys, nvarys]);
    assert_eq!(correlation[[0, 0]], 1.0); // Diagonal elements should be 1.0
    assert_eq!(correlation[[1, 1]], 1.0);
    assert_eq!(correlation[[2, 2]], 1.0);

    // Test confidence interval calculation
    let sigmas = &[1.0, 2.0];
    let conf_intervals = calculator
        .calculate_confidence_intervals(&params, &covar, sigmas)
        .unwrap();
    assert_eq!(conf_intervals.len(), nvarys);
    assert_eq!(conf_intervals["a"].len(), sigmas.len());

    // Test F-test
    // Model 1: chisqr = 10, nfree = 5
    // Model 2: chisqr = 8, nfree = 4 (one more parameter)
    // Improvement: (10-8)/10 = 0.2
    // F = 0.2 * 4 / (5-4) = 0.8
    let f_stat = calculator.f_test(10.0, 5, 8.0, 4);
    assert!((f_stat - 0.8).abs() < 1e-10);

    // Test with equal degrees of freedom (invalid case)
    let f_stat = calculator.f_test(10.0, 5, 8.0, 5);
    assert_eq!(f_stat, 1.0); // Should return 1.0 for invalid case
}

#[test]
#[cfg(feature = "matrix")]
fn test_full_uncertainty_analysis() {
    let (params, jacobian, chisqr, ndata) = create_test_data();
    let sigmas = &[1.0, 2.0];

    // Run full uncertainty analysis
    let result = uncertainty_analysis(&jacobian, &params, chisqr, ndata, sigmas).unwrap();

    // Check the structure of the result
    assert_eq!(result.covariance.shape(), &[3, 3]);
    assert_eq!(result.correlation.shape(), &[3, 3]);
    assert_eq!(result.standard_errors.len(), 3);
    assert_eq!(result.confidence_intervals.len(), 3);
    assert_eq!(result.chisqr, chisqr);
    assert_eq!(result.nfree, ndata - params.varying().len());
    assert_eq!(
        result.redchi,
        chisqr / (ndata - params.varying().len()) as f64
    );
    assert!(result.monte_carlo.is_none());

    // Check that standard errors were calculated
    for param_name in ["a", "b", "c"].iter() {
        assert!(result.standard_errors.contains_key(*param_name));
        assert!(result.standard_errors[*param_name] > 0.0);
    }

    // Check that confidence intervals were calculated
    for param_name in ["a", "b", "c"].iter() {
        assert!(result.confidence_intervals.contains_key(*param_name));
        assert_eq!(result.confidence_intervals[*param_name].len(), sigmas.len());
    }
}

#[test]
#[cfg(feature = "matrix")]
fn test_uncertainty_analysis_with_monte_carlo() {
    let (params, jacobian, chisqr, ndata) = create_test_data();
    let sigmas = &[1.0, 2.0];
    let n_samples = 50; // Small number for test speed
    let percentiles = &[0.68, 0.95];

    // Create a seeded RNG for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Run uncertainty analysis with Monte Carlo
    let result = uncertainty_analysis_with_monte_carlo(
        &jacobian,
        &params,
        chisqr,
        ndata,
        sigmas,
        n_samples,
        percentiles,
        &mut rng,
    )
    .unwrap();

    // Check the structure of the result
    assert_eq!(result.covariance.shape(), &[3, 3]);
    assert_eq!(result.correlation.shape(), &[3, 3]);
    assert_eq!(result.standard_errors.len(), 3);
    assert_eq!(result.confidence_intervals.len(), 3);
    assert_eq!(result.chisqr, chisqr);
    assert_eq!(result.nfree, ndata - params.varying().len());
    assert_eq!(
        result.redchi,
        chisqr / (ndata - params.varying().len()) as f64
    );

    // Check that Monte Carlo results exist
    assert!(result.monte_carlo.is_some());
    let mc_result = result.monte_carlo.unwrap();

    // Check Monte Carlo results
    assert_eq!(mc_result.parameter_sets.len(), n_samples);
    assert_eq!(mc_result.sorted_values.len(), 3);
    assert_eq!(mc_result.means.len(), 3);
    assert_eq!(mc_result.stds.len(), 3);
    assert_eq!(mc_result.medians.len(), 3);
    assert_eq!(mc_result.percentiles.len(), 3);

    // Check that percentiles match what we requested
    for param_name in ["a", "b", "c"].iter() {
        assert_eq!(mc_result.percentiles[*param_name].len(), percentiles.len());
        assert!((mc_result.percentiles[*param_name][0].0 - 0.68).abs() < 1e-10);
        assert!((mc_result.percentiles[*param_name][1].0 - 0.95).abs() < 1e-10);
    }
}

#[test]
#[cfg(feature = "matrix")]
fn test_uncertainty_calculator_error_cases() {
    // Test with invalid inputs
    // Case 1: ndata < nvarys (underdetermined system)
    let ndata = 2;
    let nvarys = 3;
    let chisqr = 1.0;

    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);
    assert_eq!(calculator.nfree, 1); // Should default to 1 to avoid division by zero

    // Case 2: Invalid Jacobian (rank-deficient)
    let (params, _, chisqr, ndata) = create_test_data();
    let nvarys = params.varying().len();

    let calculator = UncertaintyCalculator::new(ndata, nvarys, chisqr);

    // Create a Jacobian with linearly dependent columns
    let bad_jacobian = arr2(&[
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0], // 2 * row 1
        [3.0, 6.0, 9.0], // 3 * row 1
    ]);

    // This shouldn't crash, but will likely return an approximate covariance
    let covar = calculator.calculate_covariance(&bad_jacobian).unwrap();
    assert_eq!(covar.shape(), &[3, 3]);
}
