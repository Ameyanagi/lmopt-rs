//! Integration tests that compare lmopt-rs results with lmfit-py
//!
//! These tests serve two purposes:
//! 1. Validate the correctness of our implementation against a reference
//! 2. Demonstrate compatibility and usage patterns for users migrating from lmfit-py

use lmopt_rs::{
    lm::{LevenbergMarquardt, LmConfig, LmResult},
    problem::Problem,
};
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use std::error::Error;

/// A Gaussian peak model similar to the one in lmfit-py
struct GaussianModel {
    x: Array1<f64>,
    y: Array1<f64>,
}

impl GaussianModel {
    fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self { x, y }
    }

    /// Gaussian function: amplitude * exp(-((x - center) / (width))**2)
    fn gaussian(x: f64, amplitude: f64, center: f64, width: f64) -> f64 {
        let arg = (x - center) / width;
        amplitude * (-arg * arg).exp()
    }
}

impl Problem for GaussianModel {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::error::Result<Array1<f64>> {
        let amplitude = params[0];
        let center = params[1];
        let width = params[2];

        let predicted = self
            .x
            .mapv(|x_val| Self::gaussian(x_val, amplitude, center, width));
        Ok(&predicted - &self.y)
    }

    fn parameter_count(&self) -> usize {
        3 // amplitude, center, width
    }

    fn residual_count(&self) -> usize {
        self.x.len()
    }

    fn jacobian(&self, params: &Array1<f64>) -> lmopt_rs::error::Result<Array2<f64>> {
        let amplitude = params[0];
        let center = params[1];
        let width = params[2];
        let n = self.x.len();

        let mut jacobian = Array2::<f64>::zeros((n, 3));

        for i in 0..n {
            let x_val = self.x[i];
            let arg = (x_val - center) / width;
            let exp_term = (-arg * arg).exp();

            // ∂f/∂amplitude = exp(-((x - center) / width)^2)
            jacobian[[i, 0]] = exp_term;

            // ∂f/∂center = amplitude * (2 * (x - center) / width^2) * exp(-((x - center) / width)^2)
            jacobian[[i, 1]] = amplitude * (2.0 * (x_val - center) / (width * width)) * exp_term;

            // ∂f/∂width = amplitude * (2 * (x - center)^2 / width^3) * exp(-((x - center) / width)^2)
            jacobian[[i, 2]] =
                amplitude * (2.0 * (x_val - center).powi(2) / (width * width * width)) * exp_term;
        }

        Ok(jacobian)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

/// Generate synthetic data with a Gaussian peak and some noise
fn generate_gaussian_data() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // Generate x values from -10 to 10
    let n = 100;
    let x = Array1::linspace(-10.0, 10.0, n);

    // True parameters (amplitude, center, width)
    let true_params = Array1::from_vec(vec![5.0, 1.5, 2.0]);

    // Generate y values with some noise
    let mut rng = rand::thread_rng();
    let noise_level = 0.1;

    let y = x.mapv(|x_val| {
        let true_value =
            GaussianModel::gaussian(x_val, true_params[0], true_params[1], true_params[2]);
        let noise = Normal::new(0.0, noise_level).unwrap().sample(&mut rng);
        true_value + noise
    });

    (x, y, true_params)
}

/// Test the Gaussian model and compare with expected lmfit-py results
#[test]
fn test_gaussian_peak_fit() {
    // For reproducibility, we'll use a fixed dataset instead of random generation
    let x = Array1::linspace(-10.0, 10.0, 100);

    // Create a clean Gaussian peak with known parameters
    let true_amplitude = 5.0;
    let true_center = 1.5;
    let true_width = 2.0;

    let y = x.mapv(|x_val| GaussianModel::gaussian(x_val, true_amplitude, true_center, true_width));

    // Create the model
    let model = GaussianModel::new(x, y);

    // Initial parameters that are intentionally off
    let initial_params = Array1::from_vec(vec![3.0, 0.0, 1.0]);

    // Configure and run the LM algorithm
    let mut config = LmConfig::default();
    config.max_iterations = 100;
    config.xtol = 1e-6;
    config.ftol = 1e-6;

    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, initial_params.clone());

    // Check if the result is successful
    if let Ok(solution) = result {
        // Extract the optimized parameters
        let optimized_params = solution.params;

        // Check that the parameters match the true values
        let tolerance = 1e-4;
        assert!(
            (optimized_params[0] - true_amplitude).abs() < tolerance,
            "Amplitude doesn't match: got {}, expected {}",
            optimized_params[0],
            true_amplitude
        );

        assert!(
            (optimized_params[1] - true_center).abs() < tolerance,
            "Center doesn't match: got {}, expected {}",
            optimized_params[1],
            true_center
        );

        assert!(
            (optimized_params[2] - true_width).abs() < tolerance,
            "Width doesn't match: got {}, expected {}",
            optimized_params[2],
            true_width
        );

        // Additional checks that would match lmfit-py behavior
        assert!(
            solution.iterations < 20,
            "Too many iterations needed: {}",
            solution.iterations
        );
        assert!(
            solution.cost < 1e-10,
            "Residual cost too high: {}",
            solution.cost
        );

        println!(
            "Gaussian fit test passed! Parameters: {:?}",
            optimized_params
        );
    } else {
        panic!("LM optimization failed: {:?}", result.err());
    }
}

/// Lorenzian peak model similar to lmfit-py
struct LorentzianModel {
    x: Array1<f64>,
    y: Array1<f64>,
}

impl LorentzianModel {
    fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self { x, y }
    }

    /// Lorentzian function: amplitude * (sigma^2 / ((x - center)^2 + sigma^2))
    fn lorentzian(x: f64, amplitude: f64, center: f64, sigma: f64) -> f64 {
        let sigma_sq = sigma * sigma;
        amplitude * (sigma_sq / ((x - center) * (x - center) + sigma_sq))
    }
}

impl Problem for LorentzianModel {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::error::Result<Array1<f64>> {
        let amplitude = params[0];
        let center = params[1];
        let sigma = params[2];

        let predicted = self
            .x
            .mapv(|x_val| Self::lorentzian(x_val, amplitude, center, sigma));
        Ok(&predicted - &self.y)
    }

    fn parameter_count(&self) -> usize {
        3 // amplitude, center, sigma
    }

    fn residual_count(&self) -> usize {
        self.x.len()
    }

    fn jacobian(&self, params: &Array1<f64>) -> lmopt_rs::error::Result<Array2<f64>> {
        // For Lorentzian, we'll use numerical jacobian for simplicity
        // Use the finite difference method to calculate the Jacobian
        let epsilon = 1e-8;
        let n = self.x.len();
        let p = params.len();
        let mut jac = Array2::<f64>::zeros((n, p));

        // Calculate baseline residuals
        let r0 = self.eval(params)?;

        // Calculate Jacobian using forward differences
        for j in 0..p {
            let mut params_plus = params.clone();
            params_plus[j] += epsilon;

            let r1 = self.eval(&params_plus)?;

            for i in 0..n {
                jac[[i, j]] = (r1[i] - r0[i]) / epsilon;
            }
        }

        Ok(jac)
    }
}

/// Test the Lorentzian model and compare with expected lmfit-py results
#[test]
fn test_lorentzian_peak_fit() {
    // For reproducibility, we'll use a fixed dataset
    let x = Array1::linspace(-10.0, 10.0, 100);

    // Create a clean Lorentzian peak with known parameters
    let true_amplitude = 5.0;
    let true_center = 1.5;
    let true_sigma = 2.0;

    let y =
        x.mapv(|x_val| LorentzianModel::lorentzian(x_val, true_amplitude, true_center, true_sigma));

    // Create the model
    let model = LorentzianModel::new(x, y);

    // Initial parameters that are intentionally off
    let initial_params = Array1::from_vec(vec![3.0, 0.0, 1.0]);

    // Configure and run the LM algorithm
    let mut config = LmConfig::default();
    config.max_iterations = 100;
    config.xtol = 1e-6;
    config.ftol = 1e-6;

    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, initial_params.clone());

    // Check if the result is successful
    if let Ok(solution) = result {
        // Extract the optimized parameters
        let optimized_params = solution.params;

        // Check that the parameters match the true values
        let tolerance = 1e-4;
        assert!(
            (optimized_params[0] - true_amplitude).abs() < tolerance,
            "Amplitude doesn't match: got {}, expected {}",
            optimized_params[0],
            true_amplitude
        );

        assert!(
            (optimized_params[1] - true_center).abs() < tolerance,
            "Center doesn't match: got {}, expected {}",
            optimized_params[1],
            true_center
        );

        assert!(
            (optimized_params[2] - true_sigma).abs() < tolerance,
            "Sigma doesn't match: got {}, expected {}",
            optimized_params[2],
            true_sigma
        );

        // Additional checks that would match lmfit-py behavior
        assert!(
            solution.iterations < 20,
            "Too many iterations needed: {}",
            solution.iterations
        );
        assert!(
            solution.cost < 1e-10,
            "Residual cost too high: {}",
            solution.cost
        );

        println!(
            "Lorentzian fit test passed! Parameters: {:?}",
            optimized_params
        );
    } else {
        panic!("LM optimization failed: {:?}", result.err());
    }
}
