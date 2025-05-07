//! Tests for Monte Carlo uncertainty estimation methods
//!
//! This file contains comprehensive tests for the Monte Carlo uncertainty estimation
//! functionality in the lmopt_rs crate, focusing on:
//! - Monte Carlo analysis using covariance matrix
//! - Monte Carlo analysis using parameter refitting
//! - Uncertainty propagation to derived quantities
//! - Error handling and edge cases

use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::GlobalOptimizer;
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::problem::Problem;
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use lmopt_rs::uncertainty::{
    monte_carlo_covariance, monte_carlo_refit, propagate_uncertainty, MonteCarloResult,
    UncertaintyCalculator,
};
use ndarray::{arr1, arr2, Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Create a simple quadratic model for testing
#[derive(Clone)]
struct TestQuadraticModel {
    x: Vec<f64>,
    y: Vec<f64>,
    parameters: Parameters,
}

impl TestQuadraticModel {
    fn new(x: Vec<f64>, y: Vec<f64>, parameters: Parameters) -> Self {
        Self { x, y, parameters }
    }

    // Method to generate synthetic data for testing
    fn generate_synthetic_data(
        x_values: Vec<f64>,
        a: f64,
        b: f64,
        c: f64,
        noise_std: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, noise_std).unwrap();

        let y_values = x_values
            .iter()
            .map(|&x| {
                let y_true = a * x * x + b * x + c;
                let noise = normal.sample(&mut rng);
                y_true + noise
            })
            .collect();

        (x_values, y_values)
    }

    // Method to update the data for Monte Carlo refitting
    fn set_data(&mut self, new_y: &Array1<f64>) -> Result<()> {
        if new_y.len() != self.y.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} data points, got {}",
                self.y.len(),
                new_y.len()
            )));
        }

        // Update the y values with the new synthetic data
        for i in 0..self.y.len() {
            self.y[i] = new_y[i];
        }

        Ok(())
    }

    // Additional method for evaluating model predictions (not part of the trait)
    fn evaluate_model(&self, params: &Parameters) -> Result<Array1<f64>> {
        let a = params
            .get("a")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'a' not found".to_string()))?
            .value();
        let b = params
            .get("b")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'b' not found".to_string()))?
            .value();
        let c = params
            .get("c")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'c' not found".to_string()))?
            .value();

        // Calculate model predictions
        let mut predictions = Array1::zeros(self.x.len());

        for (i, &x) in self.x.iter().enumerate() {
            predictions[i] = a * x * x + b * x + c;
        }

        Ok(predictions)
    }
}

impl ParameterProblem for TestQuadraticModel {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        let a = self
            .parameters
            .get("a")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'a' not found".to_string()))?
            .value();
        let b = self
            .parameters
            .get("b")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'b' not found".to_string()))?
            .value();
        let c = self
            .parameters
            .get("c")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'c' not found".to_string()))?
            .value();

        // Calculate residuals: y_observed - y_model
        let mut residuals = Array1::zeros(self.x.len());

        for (i, &x) in self.x.iter().enumerate() {
            let y_model = a * x * x + b * x + c;
            residuals[i] = self.y[i] - y_model;
        }

        Ok(residuals)
    }

    fn jacobian_with_parameters(&self) -> Result<Array2<f64>> {
        // Calculate Jacobian matrix
        // Each row corresponds to a data point
        // Each column corresponds to a parameter: [∂r/∂a, ∂r/∂b, ∂r/∂c]
        // For residual = y_observed - (a*x^2 + b*x + c)
        let mut jacobian = Array2::zeros((self.x.len(), 3));

        for (i, &x) in self.x.iter().enumerate() {
            // ∂r/∂a = -x²
            jacobian[[i, 0]] = -x * x;

            // ∂r/∂b = -x
            jacobian[[i, 1]] = -x;

            // ∂r/∂c = -1
            jacobian[[i, 2]] = -1.0;
        }

        Ok(jacobian)
    }

    fn residual_count(&self) -> usize {
        self.x.len()
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

impl GlobalOptimizer for TestQuadraticModel {
    fn optimize<P: Problem>(
        &self,
        _problem: &P,
        _bounds: &[(f64, f64)],
        _max_iterations: usize,
        _max_no_improvement: usize,
        _tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        // This is just a stub implementation for testing
        Err(LmOptError::NotImplemented(
            "Custom global optimization not implemented for this example".to_string(),
        ))
    }

    fn optimize_param_problem(
        &mut self,
        max_iterations: usize,
        _max_no_improvement: usize,
        tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        // For testing, use the standard LevenbergMarquardt optimizer
        let adapter = problem_from_parameter_problem(self);
        let optimizer = lmopt_rs::lm::LevenbergMarquardt::with_default_config();
        let initial_params = self.parameters_to_array()?;

        let result = optimizer.minimize(&adapter, initial_params)?;

        // Update the parameters
        self.update_parameters_from_array(&result.params)?;

        // Convert to a GlobalOptResult
        Ok(lmopt_rs::global_opt::GlobalOptResult {
            params: result.params.clone(),
            cost: result.cost,
            iterations: result.iterations,
            func_evals: result.iterations, // Using iterations as a proxy
            success: result.success,
            message: result.message.clone(),
            local_result: Some(result),
        })
    }
}

// Helper function to create test parameters for the quadratic model
fn create_test_parameters() -> Parameters {
    let mut params = Parameters::new();
    let mut a = Parameter::new("a", 1.0); // Initial guess
    let mut b = Parameter::new("b", 1.0);
    let mut c = Parameter::new("c", 0.0);

    // Set parameters to vary
    a.set_vary(true).unwrap();
    b.set_vary(true).unwrap();
    c.set_vary(true).unwrap();

    params.add(a).unwrap();
    params.add(b).unwrap();
    params.add(c).unwrap();

    params
}

#[test]
#[cfg(feature = "matrix")]
fn test_monte_carlo_covariance() {
    // True parameter values for the quadratic model: y = 2x² + 5x + 1
    let true_a = 2.0;
    let true_b = 5.0;
    let true_c = 1.0;

    // Generate synthetic data with controlled random noise
    let x_values: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
    let (x_values, y_values) = TestQuadraticModel::generate_synthetic_data(
        x_values, true_a, true_b, true_c, 1.0, // Noise standard deviation
        42,  // Seed for reproducibility
    );

    // Create parameters with initial guesses
    let params = create_test_parameters();

    // Create the model
    let model = TestQuadraticModel::new(x_values, y_values, params);

    // Create optimization adapter
    let adapter = problem_from_parameter_problem(&model);

    // Optimize model parameters
    let optimizer = lmopt_rs::lm::LevenbergMarquardt::with_default_config();
    let initial_params = model.parameters_to_array().unwrap();
    let result = optimizer.minimize(&adapter, initial_params).unwrap();
    let mut optimized_model = model.clone();
    optimized_model
        .update_parameters_from_array(&result.params)
        .unwrap();

    // Calculate Jacobian at the optimum
    let jacobian = optimized_model.jacobian_with_parameters().unwrap();

    // Get covariance matrix
    let ndata = optimized_model.residual_count();
    let nvarys = optimized_model.parameters().varying().len();
    let covar =
        lmopt_rs::uncertainty::covariance_matrix(&jacobian, result.cost, ndata, nvarys).unwrap();

    // Test monte_carlo_covariance function
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let n_samples = 500;
    let percentiles = &[0.68, 0.95];

    let mc_result = monte_carlo_covariance(
        optimized_model.parameters(),
        &covar,
        n_samples,
        percentiles,
        &mut rng,
    )
    .unwrap();

    // Check that the result has the expected structure
    assert_eq!(mc_result.parameter_sets.len(), n_samples);
    assert!(mc_result.means.contains_key("a"));
    assert!(mc_result.means.contains_key("b"));
    assert!(mc_result.means.contains_key("c"));
    assert!(mc_result.stds.contains_key("a"));
    assert!(mc_result.stds.contains_key("b"));
    assert!(mc_result.stds.contains_key("c"));
    assert!(mc_result.medians.contains_key("a"));
    assert!(mc_result.medians.contains_key("b"));
    assert!(mc_result.medians.contains_key("c"));

    // Check that percentiles are calculated correctly
    assert_eq!(mc_result.percentiles["a"].len(), 2); // Two confidence levels
    assert_eq!(mc_result.percentiles["b"].len(), 2);
    assert_eq!(mc_result.percentiles["c"].len(), 2);

    // Check that the mean values are close to the optimized parameter values
    let a_opt = optimized_model.parameters().get("a").unwrap().value();
    let b_opt = optimized_model.parameters().get("b").unwrap().value();
    let c_opt = optimized_model.parameters().get("c").unwrap().value();

    assert!((mc_result.means["a"] - a_opt).abs() < 0.5);
    assert!((mc_result.means["b"] - b_opt).abs() < 0.5);
    assert!((mc_result.means["c"] - c_opt).abs() < 0.5);

    // Check that the confidence intervals make sense
    for param_name in ["a", "b", "c"] {
        // 68% interval should be narrower than 95% interval
        let ci_68 = &mc_result.percentiles[param_name][0];
        let ci_95 = &mc_result.percentiles[param_name][1];

        assert_eq!(ci_68.0, 0.68);
        assert_eq!(ci_95.0, 0.95);

        let width_68 = ci_68.1 .1 - ci_68.1 .0;
        let width_95 = ci_95.1 .1 - ci_95.1 .0;

        assert!(width_95 > width_68, "95% CI should be wider than 68% CI");

        // Parameter value should be within the 95% CI
        let param_value = optimized_model
            .parameters()
            .get(param_name)
            .unwrap()
            .value();
        assert!(
            param_value >= ci_95.1 .0 && param_value <= ci_95.1 .1,
            "Parameter value should be within 95% CI"
        );
    }
}

#[test]
fn test_monte_carlo_refit() {
    // True parameter values for the quadratic model: y = 2x² + 5x + 1
    let true_a = 2.0;
    let true_b = 5.0;
    let true_c = 1.0;

    // Generate synthetic data with controlled random noise
    let x_values: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
    let (x_values, y_values) = TestQuadraticModel::generate_synthetic_data(
        x_values, true_a, true_b, true_c, 1.0, // Noise standard deviation
        42,  // Seed for reproducibility
    );

    // Create parameters with initial guesses
    let params = create_test_parameters();

    // Create the model
    let mut model = TestQuadraticModel::new(x_values, y_values, params);

    // Optimize model parameters
    let result = model.optimize_param_problem(1000, 100, 1e-6).unwrap();

    // Get residuals for the best fit
    let residuals = model.eval_with_parameters().unwrap();

    // Run Monte Carlo refitting with fewer samples for test speed
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let n_samples = 20; // Small number for test speed
    let percentiles = &[0.68, 0.95];

    let mc_result = monte_carlo_refit(
        &mut model,
        model.parameters(),
        &residuals,
        n_samples,
        percentiles,
        &mut rng,
    );

    // Ensure the function at least runs without error
    assert!(mc_result.is_ok(), "monte_carlo_refit should not fail");

    if let Ok(result) = mc_result {
        // Check that the result has the expected structure
        assert!(result.parameter_sets.len() <= n_samples); // May be less if some fits failed
        assert!(result.means.contains_key("a"));
        assert!(result.means.contains_key("b"));
        assert!(result.means.contains_key("c"));

        // Check that the mean values are at least somewhat close to the true values
        // (we use a large tolerance because with only 20 samples, there can be significant variation)
        assert!((result.means["a"] - true_a).abs() < 1.0);
        assert!((result.means["b"] - true_b).abs() < 1.0);
        assert!((result.means["c"] - true_c).abs() < 1.0);
    }
}

#[test]
fn test_propagate_uncertainty() {
    // Create a sample MonteCarloResult
    let mut parameter_sets = Vec::new();
    let mut a_values = Vec::new();
    let mut b_values = Vec::new();
    let mut c_values = Vec::new();

    // Use a fixed seed for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let normal_a = Normal::new(2.0, 0.2).unwrap(); // Mean 2.0, std 0.2
    let normal_b = Normal::new(5.0, 0.5).unwrap(); // Mean 5.0, std 0.5
    let normal_c = Normal::new(1.0, 0.1).unwrap(); // Mean 1.0, std 0.1

    // Generate 1000 parameter sets
    for _ in 0..1000 {
        let a = normal_a.sample(&mut rng);
        let b = normal_b.sample(&mut rng);
        let c = normal_c.sample(&mut rng);

        let mut params = HashMap::new();
        params.insert("a".to_string(), a);
        params.insert("b".to_string(), b);
        params.insert("c".to_string(), c);

        parameter_sets.push(params);
        a_values.push(a);
        b_values.push(b);
        c_values.push(c);
    }

    // Sort values for percentiles
    a_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    b_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    c_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate means and standard deviations
    let a_mean = a_values.iter().sum::<f64>() / a_values.len() as f64;
    let b_mean = b_values.iter().sum::<f64>() / b_values.len() as f64;
    let c_mean = c_values.iter().sum::<f64>() / c_values.len() as f64;

    let a_var = a_values.iter().map(|&x| (x - a_mean).powi(2)).sum::<f64>() / a_values.len() as f64;
    let b_var = b_values.iter().map(|&x| (x - b_mean).powi(2)).sum::<f64>() / b_values.len() as f64;
    let c_var = c_values.iter().map(|&x| (x - c_mean).powi(2)).sum::<f64>() / c_values.len() as f64;

    let a_std = a_var.sqrt();
    let b_std = b_var.sqrt();
    let c_std = c_var.sqrt();

    // Calculate medians
    let a_median = a_values[a_values.len() / 2];
    let b_median = b_values[b_values.len() / 2];
    let c_median = c_values[c_values.len() / 2];

    let mut sorted_values = HashMap::new();
    sorted_values.insert("a".to_string(), a_values);
    sorted_values.insert("b".to_string(), b_values);
    sorted_values.insert("c".to_string(), c_values);

    let mut means = HashMap::new();
    means.insert("a".to_string(), a_mean);
    means.insert("b".to_string(), b_mean);
    means.insert("c".to_string(), c_mean);

    let mut stds = HashMap::new();
    stds.insert("a".to_string(), a_std);
    stds.insert("b".to_string(), b_std);
    stds.insert("c".to_string(), c_std);

    let mut medians = HashMap::new();
    medians.insert("a".to_string(), a_median);
    medians.insert("b".to_string(), b_median);
    medians.insert("c".to_string(), c_median);

    // Create dummy percentiles (not used in this test)
    let percentiles = HashMap::new();

    // Create the Monte Carlo result
    let mc_result = MonteCarloResult {
        parameter_sets,
        sorted_values,
        means,
        stds,
        medians,
        percentiles,
    };

    // Define functions for derived quantities
    // 1. Linear combination: 2*a + 3*b + c
    let linear_func =
        |params: &HashMap<String, f64>| 2.0 * params["a"] + 3.0 * params["b"] + params["c"];

    // 2. Nonlinear function: a*b^2 + c
    let nonlinear_func =
        |params: &HashMap<String, f64>| params["a"] * params["b"].powi(2) + params["c"];

    // Test linear propagation
    let percentiles = &[0.68, 0.95, 0.99];
    let (linear_values, linear_mean, linear_std, linear_median, linear_percentiles) =
        propagate_uncertainty(&mc_result, linear_func, percentiles);

    // Expected mean for linear function: 2*E[a] + 3*E[b] + E[c] = 2*2 + 3*5 + 1 = 20
    assert!(
        (linear_mean - 20.0).abs() < 0.5,
        "Linear mean should be close to 20"
    );

    // Expected std for linear function: sqrt((2*std_a)^2 + (3*std_b)^2 + std_c^2)
    // = sqrt(4*0.2^2 + 9*0.5^2 + 0.1^2) = sqrt(0.16 + 2.25 + 0.01) = sqrt(2.42) ≈ 1.56
    assert!(
        (linear_std - 1.56).abs() < 0.2,
        "Linear std should be close to 1.56"
    );

    // Check percentiles
    assert_eq!(linear_percentiles.len(), 3);
    assert!((linear_percentiles[0].0 - 0.68).abs() < 1e-10);
    assert!((linear_percentiles[1].0 - 0.95).abs() < 1e-10);
    assert!((linear_percentiles[2].0 - 0.99).abs() < 1e-10);

    // Test nonlinear propagation
    let (nonlinear_values, nonlinear_mean, nonlinear_std, nonlinear_median, nonlinear_percentiles) =
        propagate_uncertainty(&mc_result, nonlinear_func, percentiles);

    // For nonlinear function, we expect:
    // E[a*b^2 + c] ≈ E[a]*E[b^2] + E[c] = 2*(5^2 + 0.5^2) + 1 = 2*(25 + 0.25) + 1 = 2*25.25 + 1 = 51.5
    // But there will be some difference due to correlation and non-linearity
    assert!(
        (nonlinear_mean - 51.5).abs() < 3.0,
        "Nonlinear mean should be approximately 51.5"
    );

    // The percentiles should be in ascending order
    assert!(
        nonlinear_percentiles[0].1 .0 < nonlinear_percentiles[0].1 .1,
        "Lower bound should be less than upper bound"
    );
    assert!(
        nonlinear_percentiles[0].1 .1 < nonlinear_percentiles[1].1 .1,
        "95% upper bound should be greater than 68% upper bound"
    );
    assert!(
        nonlinear_percentiles[1].1 .1 < nonlinear_percentiles[2].1 .1,
        "99% upper bound should be greater than 95% upper bound"
    );
}

#[test]
fn test_uncertainty_propagation_to_model_prediction() {
    // Create parameter distribution
    let mut parameter_sets = Vec::new();

    // Use a fixed seed
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Normal distributions for parameters
    let normal_a = Normal::new(2.0, 0.2).unwrap();
    let normal_b = Normal::new(5.0, 0.5).unwrap();
    let normal_c = Normal::new(1.0, 0.1).unwrap();

    // Generate parameters
    for _ in 0..1000 {
        let a = normal_a.sample(&mut rng);
        let b = normal_b.sample(&mut rng);
        let c = normal_c.sample(&mut rng);

        let mut params = HashMap::new();
        params.insert("a".to_string(), a);
        params.insert("b".to_string(), b);
        params.insert("c".to_string(), c);

        parameter_sets.push(params);
    }

    // Create a dummy MonteCarloResult
    // (We only need parameter_sets, the rest is unused in this test)
    let dummy_mc_result = MonteCarloResult {
        parameter_sets,
        sorted_values: HashMap::new(),
        means: HashMap::new(),
        stds: HashMap::new(),
        medians: HashMap::new(),
        percentiles: HashMap::new(),
    };

    // Define x points where we want to evaluate the model
    let x_values = vec![-3.0, -1.0, 0.0, 1.0, 3.0];

    // For each x value, create a function to evaluate the model and propagate uncertainty
    let percentiles = &[0.68, 0.95];
    let mut predictions = HashMap::new();

    for &x in &x_values {
        // Define a function to evaluate the quadratic at this x
        let model_at_x = move |params: &HashMap<String, f64>| {
            let a = params["a"];
            let b = params["b"];
            let c = params["c"];
            a * x * x + b * x + c
        };

        // Propagate uncertainty
        let (values, mean, std_dev, median, percs) =
            propagate_uncertainty(&dummy_mc_result, model_at_x, percentiles);

        predictions.insert(x, (mean, std_dev, percs));
    }

    // Check predictions at x = 0
    // Should be just the value of parameter c
    let pred_at_0 = &predictions[&0.0];
    assert!((pred_at_0.0 - 1.0).abs() < 0.2); // Mean should be close to c=1.0
    assert!((pred_at_0.1 - 0.1).abs() < 0.05); // Std dev should be close to std_c=0.1

    // Check predictions at x = 1
    // Should be a + b + c = 2 + 5 + 1 = 8
    let pred_at_1 = &predictions[&1.0];
    assert!((pred_at_1.0 - 8.0).abs() < 0.5);

    // Check predictions at x = 3
    // Should be 9*a + 3*b + c = 9*2 + 3*5 + 1 = 18 + 15 + 1 = 34
    let pred_at_3 = &predictions[&3.0];
    assert!((pred_at_3.0 - 34.0).abs() < 1.0);

    // Check that uncertainty increases with the magnitude of x
    // This is because the coefficient of x^2 (a) has uncertainty
    assert!(
        predictions[&0.0].1 < predictions[&1.0].1 && predictions[&1.0].1 < predictions[&3.0].1,
        "Uncertainty should increase with |x|"
    );
}

#[test]
fn test_monte_carlo_covariance_error_cases() {
    // Test with empty parameter set
    let empty_params = Parameters::new();
    let covar = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let result = monte_carlo_covariance(&empty_params, &covar, 100, &[0.68], &mut rng);
    assert!(result.is_err());

    // Test with mismatched covariance matrix dimensions
    let mut params = Parameters::new();
    let mut p1 = Parameter::new("a", 1.0);
    p1.set_vary(true).unwrap();
    params.add(p1).unwrap();

    // Covariance is 2x2 but only one varying parameter
    let result = monte_carlo_covariance(&params, &covar, 100, &[0.68], &mut rng);
    assert!(result.is_err());
}

#[test]
fn test_monte_carlo_refit_error_cases() {
    // Create a simple model
    let x_values = vec![1.0, 2.0, 3.0];
    let y_values = vec![1.0, 4.0, 9.0]; // Simple quadratic
    let mut params = Parameters::new();
    let mut p1 = Parameter::new("a", 1.0);
    p1.set_vary(true).unwrap();
    params.add(p1).unwrap();

    let mut model = TestQuadraticModel::new(x_values, y_values, params);
    let residuals = arr1(&[0.0, 0.0, 0.0]); // Perfect fit
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Test with empty percentiles
    let result = monte_carlo_refit(
        &mut model,
        &model.parameters(),
        &residuals,
        10,
        &[],
        &mut rng,
    );
    // Should work but return no percentiles
    assert!(result.is_ok());
    if let Ok(mc_result) = result {
        assert!(mc_result.percentiles["a"].is_empty());
    }

    // Test with wrong residual length
    let wrong_residuals = arr1(&[0.0, 0.0]); // Too short
    let result = monte_carlo_refit(
        &mut model,
        &model.parameters(),
        &wrong_residuals,
        10,
        &[0.68],
        &mut rng,
    );
    assert!(result.is_err());
}
