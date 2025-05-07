//! Tests for confidence interval calculation methods
//!
//! This file contains comprehensive tests for the confidence interval
//! calculation functionality in the lmopt_rs crate, focusing on:
//! - Basic confidence interval calculations from covariance matrix
//! - Confidence region calculations for parameter pairs
//! - Probability to sigma conversion and vice versa
//! - Error handling

use lmopt_rs::error::Result;
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::uncertainty::{
    calculate_correlation, calculate_covariance, confidence_intervals, confidence_regions_2d,
    probability_to_sigma, sigma_to_probability, standard_errors_from_covariance,
    ConfidenceInterval,
};
use ndarray::{arr1, arr2, Array1, Array2};
use std::collections::HashMap;

// Helper function to create a set of test parameters for a linear model
fn create_linear_parameters() -> Parameters {
    let mut params = Parameters::new();
    let mut slope = Parameter::new("slope", 2.0);
    let mut intercept = Parameter::new("intercept", 5.0);

    slope.set_vary(true).unwrap();
    intercept.set_vary(true).unwrap();

    params.add(slope).unwrap();
    params.add(intercept).unwrap();

    params
}

// Helper function to create a covariance matrix for a linear model
fn create_linear_covariance() -> Array2<f64> {
    // Create a covariance matrix with some correlation
    // [ 0.4  0.2 ]
    // [ 0.2  0.5 ]
    arr2(&[[0.4, 0.2], [0.2, 0.5]])
}

#[test]
fn test_confidence_intervals_basic() {
    let params = create_linear_parameters();
    let covar = create_linear_covariance();

    // Calculate confidence intervals at 1, 2, and 3 sigma levels
    let sigmas = &[1.0, 2.0, 3.0];
    let intervals = confidence_intervals(&params, &covar, sigmas, None).unwrap();

    // Check that we have intervals for both parameters
    assert!(intervals.contains_key("slope"));
    assert!(intervals.contains_key("intercept"));

    // Check that we have 3 intervals for each parameter
    assert_eq!(intervals["slope"].len(), 3);
    assert_eq!(intervals["intercept"].len(), 3);

    // Check interval values (slope has std_err = sqrt(0.4) = 0.632...)
    let slope_value = params.get("slope").unwrap().value();
    let slope_std_err = 0.4f64.sqrt();

    // 1-sigma intervals for slope should be [slope-std_err, slope+std_err]
    let slope_1sigma = &intervals["slope"][0];
    assert!((slope_1sigma.probability - 0.6827).abs() < 0.01);
    assert!((slope_1sigma.lower - (slope_value - slope_std_err)).abs() < 0.01);
    assert!((slope_1sigma.upper - (slope_value + slope_std_err)).abs() < 0.01);

    // 2-sigma intervals should be wider than 1-sigma
    let slope_2sigma = &intervals["slope"][1];
    assert!(slope_2sigma.probability > slope_1sigma.probability);
    assert!(slope_2sigma.lower < slope_1sigma.lower);
    assert!(slope_2sigma.upper > slope_1sigma.upper);

    // Similar checks for intercept (std_err = sqrt(0.5) = 0.707...)
    let intercept_value = params.get("intercept").unwrap().value();
    let intercept_std_err = 0.5f64.sqrt();

    let intercept_1sigma = &intervals["intercept"][0];
    assert!((intercept_1sigma.lower - (intercept_value - intercept_std_err)).abs() < 0.01);
    assert!((intercept_1sigma.upper - (intercept_value + intercept_std_err)).abs() < 0.01);
}

#[test]
fn test_confidence_intervals_edge_cases() {
    // Test with a parameter that has zero variance
    let mut params = Parameters::new();
    let mut p1 = Parameter::new("p1", 1.0);
    let mut p2 = Parameter::new("p2", 2.0);
    p1.set_vary(true).unwrap();
    p2.set_vary(true).unwrap();
    params.add(p1).unwrap();
    params.add(p2).unwrap();

    // Covariance with zero variance for p1
    let covar = arr2(&[[0.0, 0.0], [0.0, 0.5]]);

    let sigmas = &[1.0];
    let intervals = confidence_intervals(&params, &covar, sigmas, None).unwrap();

    // p1 should still have an interval, but lower = upper = value
    let p1_interval = &intervals["p1"][0];
    assert!((p1_interval.lower - p1_interval.upper).abs() < 1e-10);

    // Test with non-varying parameters
    let mut params = Parameters::new();
    let mut p1 = Parameter::new("p1", 1.0);
    let mut p2 = Parameter::new("p2", 2.0);
    p1.set_vary(true).unwrap();
    p2.set_vary(false).unwrap(); // p2 does not vary
    params.add(p1).unwrap();
    params.add(p2).unwrap();

    // Covariance for just p1
    let covar = arr2(&[[0.4]]);

    let intervals = confidence_intervals(&params, &covar, sigmas, None).unwrap();

    // Only p1 should have intervals
    assert!(intervals.contains_key("p1"));
    assert!(!intervals.contains_key("p2"));

    // Test with custom probability function
    let prob_func = |sigma: f64, _dof: f64| {
        // Custom mapping from sigma to probability
        match sigma as i64 {
            1 => 0.7,
            2 => 0.9,
            3 => 0.99,
            _ => sigma_to_probability(sigma),
        }
    };

    let params = create_linear_parameters();
    let covar = create_linear_covariance();

    let intervals = confidence_intervals(&params, &covar, sigmas, Some(prob_func)).unwrap();

    // Check that probabilities match our custom function
    assert!((intervals["slope"][0].probability - 0.7).abs() < 0.01);
    assert!((intervals["slope"][1].probability - 0.9).abs() < 0.01);
    assert!((intervals["slope"][2].probability - 0.99).abs() < 0.01);
}

#[test]
fn test_confidence_regions_2d() {
    let params = create_linear_parameters();
    let covar = create_linear_covariance();

    // Calculate 2D confidence region
    let (x_grid, y_grid, chi2_grid) = confidence_regions_2d(
        &params,
        "slope",
        "intercept",
        10,  // x grid points
        10,  // y grid points
        2.0, // 2-sigma region
        &covar,
    )
    .unwrap();

    // Check grid dimensions
    assert_eq!(x_grid.len(), 10);
    assert_eq!(y_grid.len(), 10);
    assert_eq!(chi2_grid.shape(), &[10, 10]);

    // Check that the grid points include the parameter values
    let slope_val = params.get("slope").unwrap().value();
    let intercept_val = params.get("intercept").unwrap().value();

    assert!(x_grid[0] < slope_val && x_grid[x_grid.len() - 1] > slope_val);
    assert!(y_grid[0] < intercept_val && y_grid[y_grid.len() - 1] > intercept_val);

    // Chi-square should be minimized near the parameter values
    // Find the grid point closest to the parameter values
    let mut min_i = 0;
    let mut min_j = 0;
    let mut min_dist = f64::MAX;

    for i in 0..x_grid.len() {
        for j in 0..y_grid.len() {
            let dist = (x_grid[i] - slope_val).powi(2) + (y_grid[j] - intercept_val).powi(2);
            if dist < min_dist {
                min_dist = dist;
                min_i = i;
                min_j = j;
            }
        }
    }

    // The chi-square value at this point should be close to zero
    assert!(chi2_grid[[min_i, min_j]] < 1.0);

    // Chi-square should increase as we move away from the minimum
    for i in 0..x_grid.len() {
        for j in 0..y_grid.len() {
            if i != min_i || j != min_j {
                let dist_from_min =
                    (i as i32 - min_i as i32).pow(2) + (j as i32 - min_j as i32).pow(2);
                if dist_from_min == 1 {
                    // Adjacent point
                    assert!(chi2_grid[[i, j]] >= chi2_grid[[min_i, min_j]]);
                }
            }
        }
    }
}

#[test]
fn test_confidence_regions_2d_error_cases() {
    let params = create_linear_parameters();
    let covar = create_linear_covariance();

    // Test with non-existent parameter
    let result = confidence_regions_2d(&params, "nonexistent", "intercept", 10, 10, 2.0, &covar);
    assert!(result.is_err());

    // Test with non-varying parameter
    let mut params = Parameters::new();
    let p1 = Parameter::new("p1", 1.0);
    let mut p2 = Parameter::new("p2", 2.0);
    p2.set_vary(true).unwrap();
    params.add(p1).unwrap(); // p1 does not vary
    params.add(p2).unwrap();

    let result = confidence_regions_2d(&params, "p1", "p2", 10, 10, 2.0, &arr2(&[[0.1]]));
    assert!(result.is_err());
}

#[test]
fn test_probability_sigma_conversion() {
    // Test sigma to probability
    let p1 = sigma_to_probability(1.0);
    let p2 = sigma_to_probability(2.0);
    let p3 = sigma_to_probability(3.0);

    // Check against standard values
    assert!((p1 - 0.6827).abs() < 0.01);
    assert!((p2 - 0.9545).abs() < 0.01);
    assert!((p3 - 0.9973).abs() < 0.01);

    // Test probability to sigma
    let s1 = probability_to_sigma(0.6827);
    let s2 = probability_to_sigma(0.9545);
    let s3 = probability_to_sigma(0.9973);

    assert!((s1 - 1.0).abs() < 0.1);
    assert!((s2 - 2.0).abs() < 0.1);
    assert!((s3 - 3.0).abs() < 0.1);

    // Test roundtrip conversion
    for sigma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0].iter() {
        let prob = sigma_to_probability(*sigma);
        let sigma_back = probability_to_sigma(prob);
        assert!(
            (sigma_back - sigma).abs() < 0.2,
            "Roundtrip conversion failed for sigma = {}: {} -> {} -> {}",
            sigma,
            sigma,
            prob,
            sigma_back
        );
    }

    // Test special cases
    assert!(sigma_to_probability(0.5) == 0.5); // Values below 1 are treated as probabilities
    assert!(probability_to_sigma(1.5) == 1.5); // Values above 1 are treated as sigmas
}

#[test]
#[cfg(feature = "matrix")]
fn test_covariance_calculation() {
    // Create a test Jacobian matrix for a linear fit
    // y = a*x + b with data points (1,3), (2,5), (3,7), (4,9)
    // Jacobian = [dx, 1]
    let jacobian = arr2(&[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]);

    // Calculate covariance (reduced chi-square = 1.0)
    let covar = calculate_covariance(&jacobian, 1.0).unwrap();

    // Check dimensions
    assert_eq!(covar.shape(), &[2, 2]);

    // Covariance matrix should be symmetric
    assert!((covar[[0, 1]] - covar[[1, 0]]).abs() < 1e-10);

    // Standard errors should be sqrt of diagonal elements
    let std_errors = standard_errors_from_covariance(&covar);
    assert_eq!(std_errors.len(), 2);
    assert!((std_errors[0] - covar[[0, 0]].sqrt()).abs() < 1e-10);
    assert!((std_errors[1] - covar[[1, 1]].sqrt()).abs() < 1e-10);

    // Calculate correlation matrix
    let correl = calculate_correlation(&covar);

    // Correlation matrix should have diagonal elements = 1.0
    assert_eq!(correl[[0, 0]], 1.0);
    assert_eq!(correl[[1, 1]], 1.0);

    // Off-diagonal elements should be correlation coefficients
    let expected_corr = covar[[0, 1]] / (covar[[0, 0]] * covar[[1, 1]]).sqrt();
    assert!((correl[[0, 1]] - expected_corr).abs() < 1e-10);
    assert!((correl[[1, 0]] - expected_corr).abs() < 1e-10);

    // Correlation coefficients should be between -1 and 1
    assert!(correl[[0, 1]].abs() <= 1.0);
}
