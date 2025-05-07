//! Tests for uncertainty analysis with model fitting
//!
//! This file demonstrates how to use uncertainty analysis in the context
//! of model fitting with the lmopt_rs crate. It showcases typical workflows
//! for calculating parameter uncertainties after fitting a model to data.

use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::GlobalOptimizer;
use lmopt_rs::lm::LevenbergMarquardt;
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::problem::Problem;
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use lmopt_rs::uncertainty::{
    covariance_matrix, monte_carlo_covariance, standard_errors, uncertainty_analysis,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// A simple Gaussian peak model for testing
struct GaussianModel {
    x: Vec<f64>,            // x values
    y: Vec<f64>,            // y values (data)
    parameters: Parameters, // parameters: amplitude, center, sigma
}

impl GaussianModel {
    fn new(x: Vec<f64>, y: Vec<f64>, parameters: Parameters) -> Self {
        Self { x, y, parameters }
    }

    /// Generate synthetic Gaussian data for testing
    fn generate_synthetic_data(
        x_values: Vec<f64>,
        amplitude: f64,
        center: f64,
        sigma: f64,
        noise_std: f64,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let normal = Normal::new(0.0, noise_std).unwrap();

        let y_values = x_values
            .iter()
            .map(|&x| {
                let z = (x - center) / sigma;
                let y_true = amplitude * (-0.5 * z * z).exp();
                let noise = normal.sample(&mut rng);
                y_true + noise
            })
            .collect();

        (x_values, y_values)
    }

    /// Create a set of parameters for the Gaussian model
    fn create_parameters(amp_init: f64, center_init: f64, sigma_init: f64) -> Parameters {
        let mut params = Parameters::new();
        let mut amp = Parameter::new("amplitude", amp_init);
        let mut center = Parameter::new("center", center_init);
        let mut sigma = Parameter::new("sigma", sigma_init);

        amp.set_vary(true).unwrap();
        center.set_vary(true).unwrap();
        sigma.set_vary(true).unwrap();

        // Add bounds to prevent sigma from going negative
        sigma.set_min(0.0).unwrap();

        params.add(amp).unwrap();
        params.add(center).unwrap();
        params.add(sigma).unwrap();

        params
    }
}

/// Implement the ParameterProblem trait for GaussianModel
impl ParameterProblem for GaussianModel {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        let amplitude = self
            .parameters
            .get("amplitude")
            .ok_or_else(|| {
                LmOptError::ParameterError("Parameter 'amplitude' not found".to_string())
            })?
            .value();
        let center = self
            .parameters
            .get("center")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'center' not found".to_string()))?
            .value();
        let sigma = self
            .parameters
            .get("sigma")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'sigma' not found".to_string()))?
            .value();

        // Calculate residuals: y_observed - y_model
        let mut residuals = Array1::zeros(self.x.len());

        for (i, &x) in self.x.iter().enumerate() {
            let z = (x - center) / sigma;
            let y_model = amplitude * (-0.5 * z * z).exp();
            residuals[i] = self.y[i] - y_model;
        }

        Ok(residuals)
    }

    fn jacobian_with_parameters(&self) -> Result<Array2<f64>> {
        let amplitude = self
            .parameters
            .get("amplitude")
            .ok_or_else(|| {
                LmOptError::ParameterError("Parameter 'amplitude' not found".to_string())
            })?
            .value();
        let center = self
            .parameters
            .get("center")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'center' not found".to_string()))?
            .value();
        let sigma = self
            .parameters
            .get("sigma")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'sigma' not found".to_string()))?
            .value();

        // Calculate Jacobian matrix
        // Each row is a data point, columns are derivatives w.r.t. parameters
        // J = [∂r/∂amplitude, ∂r/∂center, ∂r/∂sigma]
        let mut jacobian = Array2::zeros((self.x.len(), 3));

        for (i, &x) in self.x.iter().enumerate() {
            let z = (x - center) / sigma;
            let exp_term = (-0.5 * z * z).exp();

            // ∂r/∂amplitude = -exp(-0.5*z^2)
            jacobian[[i, 0]] = -exp_term;

            // ∂r/∂center = -amplitude * exp(-0.5*z^2) * (x-center)/(sigma^2)
            jacobian[[i, 1]] = -amplitude * exp_term * (x - center) / (sigma * sigma);

            // ∂r/∂sigma = -amplitude * exp(-0.5*z^2) * (x-center)^2/(sigma^3)
            jacobian[[i, 2]] =
                -amplitude * exp_term * (x - center) * (x - center) / (sigma * sigma * sigma);
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

impl GlobalOptimizer for GaussianModel {
    fn optimize<P: Problem>(
        &self,
        _problem: &P,
        _bounds: &[(f64, f64)],
        _max_iterations: usize,
        _max_no_improvement: usize,
        _tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        Err(LmOptError::NotImplemented(
            "Global optimization not implemented for this example".to_string(),
        ))
    }

    fn optimize_param_problem(
        &mut self,
        max_iterations: usize,
        _max_no_improvement: usize,
        tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        let adapter = problem_from_parameter_problem(self);
        let optimizer = LevenbergMarquardt::with_default_config();
        let initial_params = self.parameters_to_array()?;

        let result = optimizer.minimize(&adapter, initial_params)?;

        // Update the parameters
        self.update_parameters_from_array(&result.params)?;

        // Convert to a GlobalOptResult
        Ok(lmopt_rs::global_opt::GlobalOptResult {
            params: result.params.clone(),
            cost: result.cost,
            iterations: result.iterations,
            func_evals: result.iterations,
            success: result.success,
            message: result.message.clone(),
            local_result: Some(result),
        })
    }
}

#[test]
#[cfg(feature = "matrix")]
fn test_model_fit_with_uncertainty() {
    // True parameter values
    let true_amplitude = 5.0;
    let true_center = 10.0;
    let true_sigma = 2.0;

    // Generate synthetic data with controlled noise
    let x_values: Vec<f64> = (0..=20).map(|i| i as f64).collect();
    let (x_values, y_values) = GaussianModel::generate_synthetic_data(
        x_values,
        true_amplitude,
        true_center,
        true_sigma,
        0.2, // Low noise for better fit
        42,  // Seed for reproducibility
    );

    // Create parameters with initial guesses
    let params = GaussianModel::create_parameters(4.0, 9.0, 1.5);

    // Create the model
    let mut model = GaussianModel::new(x_values, y_values, params);

    // Fit the model
    let result = model.optimize_param_problem(100, 10, 1e-6).unwrap();

    // Verify the fit was successful
    assert!(result.success);

    // Calculate Jacobian at the best fit
    let jacobian = model.jacobian_with_parameters().unwrap();

    // Calculate covariance matrix
    let ndata = model.residual_count();
    let nvarys = model.parameters().varying().len();
    let covar = covariance_matrix(&jacobian, result.cost, ndata, nvarys).unwrap();

    // Calculate standard errors
    let std_errors = standard_errors(&covar, model.parameters());

    // Verify we have standard errors for all parameters
    assert!(std_errors.contains_key("amplitude"));
    assert!(std_errors.contains_key("center"));
    assert!(std_errors.contains_key("sigma"));

    // Perform full uncertainty analysis
    let sigmas = &[1.0, 2.0];
    let uncertainty =
        uncertainty_analysis(&jacobian, model.parameters(), result.cost, ndata, sigmas).unwrap();

    // Check that the confidence intervals contain the true values
    let amplitude = model.parameters().get("amplitude").unwrap().value();
    let center = model.parameters().get("center").unwrap().value();
    let sigma = model.parameters().get("sigma").unwrap().value();

    // The fit should be close to the true values
    assert!((amplitude - true_amplitude).abs() < 0.5);
    assert!((center - true_center).abs() < 0.5);
    assert!((sigma - true_sigma).abs() < 0.5);

    // The 2-sigma confidence intervals should include the true values
    let amp_intervals = &uncertainty.confidence_intervals["amplitude"];
    let center_intervals = &uncertainty.confidence_intervals["center"];
    let sigma_intervals = &uncertainty.confidence_intervals["sigma"];

    // Check the 2-sigma (95%) intervals
    let amp_2sigma = &amp_intervals[1];
    let center_2sigma = &center_intervals[1];
    let sigma_2sigma = &sigma_intervals[1];

    assert!(true_amplitude >= amp_2sigma.lower && true_amplitude <= amp_2sigma.upper);
    assert!(true_center >= center_2sigma.lower && true_center <= center_2sigma.upper);
    assert!(true_sigma >= sigma_2sigma.lower && true_sigma <= sigma_2sigma.upper);

    // Run Monte Carlo analysis to get a more robust estimate
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mc_results =
        monte_carlo_covariance(model.parameters(), &covar, 1000, &[0.68, 0.95], &mut rng).unwrap();

    // Check that the Monte Carlo means are close to the best-fit values
    assert!((mc_results.means["amplitude"] - amplitude).abs() < 0.2);
    assert!((mc_results.means["center"] - center).abs() < 0.2);
    assert!((mc_results.means["sigma"] - sigma).abs() < 0.2);

    // The 95% Monte Carlo intervals should also contain the true values
    let amp_mc_95 = &mc_results.percentiles["amplitude"][1].1;
    let center_mc_95 = &mc_results.percentiles["center"][1].1;
    let sigma_mc_95 = &mc_results.percentiles["sigma"][1].1;

    assert!(true_amplitude >= amp_mc_95.0 && true_amplitude <= amp_mc_95.1);
    assert!(true_center >= center_mc_95.0 && true_center <= center_mc_95.1);
    assert!(true_sigma >= sigma_mc_95.0 && true_sigma <= sigma_mc_95.1);
}

#[test]
#[cfg(feature = "matrix")]
fn test_model_fit_with_fixed_parameter() {
    // True parameter values
    let true_amplitude = 5.0;
    let true_center = 10.0;
    let true_sigma = 2.0;

    // Generate synthetic data
    let x_values: Vec<f64> = (0..=20).map(|i| i as f64).collect();
    let (x_values, y_values) = GaussianModel::generate_synthetic_data(
        x_values,
        true_amplitude,
        true_center,
        true_sigma,
        0.2,
        42,
    );

    // Create parameters with one fixed
    let mut params = Parameters::new();
    let mut amp = Parameter::new("amplitude", 4.0);
    let mut center = Parameter::new("center", 9.0);
    let mut sigma = Parameter::new("sigma", 2.0); // Fixed at true value

    amp.set_vary(true).unwrap();
    center.set_vary(true).unwrap();
    sigma.set_vary(false).unwrap(); // Fix sigma

    params.add(amp).unwrap();
    params.add(center).unwrap();
    params.add(sigma).unwrap();

    // Create and fit the model
    let mut model = GaussianModel::new(x_values, y_values, params);
    let result = model.optimize_param_problem(100, 10, 1e-6).unwrap();
    assert!(result.success);

    // Calculate Jacobian and covariance
    let jacobian = model.jacobian_with_parameters().unwrap();
    let ndata = model.residual_count();
    let nvarys = model.parameters().varying().len();

    assert_eq!(nvarys, 2); // Only amplitude and center vary

    let covar = covariance_matrix(&jacobian, result.cost, ndata, nvarys).unwrap();

    // The covariance matrix should be 2x2 (only varying parameters)
    assert_eq!(covar.shape(), &[2, 2]);

    // Calculate standard errors
    let std_errors = standard_errors(&covar, model.parameters());

    // Only varying parameters should have standard errors
    assert!(std_errors.contains_key("amplitude"));
    assert!(std_errors.contains_key("center"));
    assert!(!std_errors.contains_key("sigma"));

    // Perform uncertainty analysis
    let uncertainty = uncertainty_analysis(
        &jacobian,
        model.parameters(),
        result.cost,
        ndata,
        &[1.0, 2.0],
    )
    .unwrap();

    // Check confidence intervals
    assert!(uncertainty.confidence_intervals.contains_key("amplitude"));
    assert!(uncertainty.confidence_intervals.contains_key("center"));
    assert!(!uncertainty.confidence_intervals.contains_key("sigma"));

    // Fitting with sigma fixed should give a better estimate of amplitude and center
    // since we've reduced the parameter space
    let amplitude = model.parameters().get("amplitude").unwrap().value();
    let center = model.parameters().get("center").unwrap().value();

    // These should be very close to the true values since sigma is fixed at the correct value
    assert!((amplitude - true_amplitude).abs() < 0.2);
    assert!((center - true_center).abs() < 0.2);
}
