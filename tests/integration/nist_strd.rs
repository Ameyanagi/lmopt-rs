//! Integration tests for the Levenberg-Marquardt algorithm using NIST Statistical
//! Reference Datasets (StRD)
//!
//! These datasets provide standard nonlinear regression problems with known solutions
//! that are challenging for many optimization methods.
//!
//! Data source: https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml

use lmopt_rs::{
    error::LmOptError,
    lm::{LevenbergMarquardt, LmConfig},
    problem::Problem,
};
use ndarray::{Array1, Array2};
use std::error::Error;

/// A generic NIST model that implements the Problem trait
struct NISTModel {
    x: Array2<f64>,
    y: Array1<f64>,
    model_fn: fn(&Array1<f64>, &Array2<f64>) -> Array1<f64>,
    jacobian_fn: Option<fn(&Array1<f64>, &Array2<f64>) -> Array2<f64>>,
}

impl NISTModel {
    fn new(
        x: Array2<f64>,
        y: Array1<f64>,
        model_fn: fn(&Array1<f64>, &Array2<f64>) -> Array1<f64>,
        jacobian_fn: Option<fn(&Array1<f64>, &Array2<f64>) -> Array2<f64>>,
    ) -> Self {
        Self {
            x,
            y,
            model_fn,
            jacobian_fn,
        }
    }
}

impl Problem for NISTModel {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>, LmOptError> {
        let predicted = (self.model_fn)(params, &self.x);
        Ok(&predicted - &self.y)
    }

    fn parameter_count(&self) -> usize {
        if self.x.shape()[1] > 0 {
            self.x.shape()[1]
        } else {
            // If x is a column vector, assume parameters match the number of
            // variables needed by the model function
            2 // Default assumption for NIST models
        }
    }

    fn residual_count(&self) -> usize {
        self.y.len()
    }

    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>, LmOptError> {
        if let Some(jacobian_fn) = self.jacobian_fn {
            Ok(jacobian_fn(params, &self.x))
        } else {
            // Fallback to numerical Jacobian using finite differences
            let n = self.y.len();
            let p = params.len();
            let mut jac = Array2::<f64>::zeros((n, p));
            let epsilon = 1e-8;

            // Compute Jacobian column by column
            for j in 0..p {
                let mut params_plus = params.clone();
                params_plus[j] += epsilon;

                let r0 = self.eval(params)?;
                let r1 = self.eval(&params_plus)?;

                // Compute derivative via forward difference
                for i in 0..n {
                    jac[[i, j]] = (r1[i] - r0[i]) / epsilon;
                }
            }

            Ok(jac)
        }
    }

    fn has_custom_jacobian(&self) -> bool {
        self.jacobian_fn.is_some()
    }
}

/// Load the Misra1a dataset and prepare the model
fn load_misra1a() -> Result<(NISTModel, Array1<f64>, Array1<f64>), Box<dyn Error>> {
    // Misra1a dataset
    // y = b1 * (1 - exp(-b2 * x))

    // For testing purposes, we'll hardcode a small subset of the data
    // In a real implementation, this would load from a file

    let x_data = vec![
        77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8, 593.1, 689.1,
        760.0,
    ];

    let y_data = vec![
        10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01, 66.40, 75.47,
        81.78,
    ];

    // Certified values from NIST
    let certified_params = Array1::from_vec(vec![2.3894212918E+02, 5.5015643181E-04]);
    let certified_std_devs = Array1::from_vec(vec![2.7070075241E+00, 1.2889360290E-05]);

    // Convert data to ndarray
    let x = Array2::from_shape_vec((x_data.len(), 1), x_data)?;
    let y = Array1::from_vec(y_data);

    // Create model function for Misra1a
    let model_fn = |params: &Array1<f64>, x: &Array2<f64>| -> Array1<f64> {
        let b1 = params[0];
        let b2 = params[1];

        x.column(0).mapv(|x_val| b1 * (1.0 - (-b2 * x_val).exp()))
    };

    // Create analytical jacobian (optional)
    let jacobian_fn = |params: &Array1<f64>, x: &Array2<f64>| -> Array2<f64> {
        let b1 = params[0];
        let b2 = params[1];
        let n = x.nrows();

        let mut jacobian = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let x_val = x[[i, 0]];
            let exp_term = (-b2 * x_val).exp();

            // ∂f/∂b1 = (1 - exp(-b2 * x))
            jacobian[[i, 0]] = 1.0 - exp_term;

            // ∂f/∂b2 = b1 * x * exp(-b2 * x)
            jacobian[[i, 1]] = b1 * x_val * exp_term;
        }

        jacobian
    };

    let model = NISTModel::new(x, y, model_fn, Some(jacobian_fn));

    Ok((model, certified_params, certified_std_devs))
}

/// Test the Levenberg-Marquardt algorithm on the Misra1a dataset
#[test]
fn test_misra1a_model() {
    let (model, certified_params, _certified_std_devs) = load_misra1a().unwrap();

    // Starting values for b1 and b2 (NIST suggests two starting points)
    let initial_params = Array1::from_vec(vec![500.0, 0.0001]);

    // Configure and run the LM algorithm
    let config = LmConfig {
        max_iterations: 1000,
        xtol: 1e-7,
        ftol: 1e-7,
        gtol: 1e-7,
        ..LmConfig::default()
    };

    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, initial_params.clone());

    // Check if the result is successful
    if let Ok(solution) = result {
        // Extract the optimized parameters
        let optimized_params = solution.params;
        let residual_norm = solution.cost;

        // Check that the parameters are close to the certified values
        for i in 0..certified_params.len() {
            let rel_error = (optimized_params[i] - certified_params[i]).abs() / certified_params[i];
            assert!(
                rel_error < 0.01,
                "Parameter {} is not close enough to certified value. Got {}, expected {}",
                i,
                optimized_params[i],
                certified_params[i]
            );
        }

        // Check that the residual norm is reasonable
        let expected_residual_norm = 1.2455138894E-01; // From NIST certification
        let rel_error = (residual_norm - expected_residual_norm).abs() / expected_residual_norm;
        assert!(
            rel_error < 0.01,
            "Residual norm is not close enough to certified value. Got {}, expected {}",
            residual_norm,
            expected_residual_norm
        );

        println!("Misra1a test passed! Parameters: {:?}", optimized_params);
    } else {
        panic!("LM optimization failed: {:?}", result.err());
    }
}

/// Load the Thurber dataset and prepare the model
fn load_thurber() -> Result<(NISTModel, Array1<f64>, Array1<f64>), Box<dyn Error>> {
    // Thurber dataset
    // Rational function: y = (b1 + b2*x + b3*x^2 + b4*x^3) / (1 + b5*x + b6*x^2 + b7*x^3)

    // For testing purposes, we'll hardcode a small subset of the data
    // In a real implementation, this would load from a file

    let x_data = vec![
        -3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699, -2.633, -2.481, -2.363,
        -2.322, -1.501, -1.460, -1.274, -1.212, -1.100, -1.046, -0.915, -0.714, -0.566, -0.545,
        -0.400, -0.309, -0.109, -0.103, 0.010, 0.119, 0.377, 0.790, 0.963, 1.006, 1.115, 1.572,
        1.841, 2.047, 2.200,
    ];

    let y_data = vec![
        80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 92.405, 95.854, 100.696,
        101.060, 401.672, 390.724, 567.534, 635.316, 733.054, 759.087, 894.206, 990.785, 1090.109,
        1080.914, 1122.643, 1178.351, 1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509,
        1425.208, 1421.384, 1442.962, 1464.350, 1468.705, 1447.894, 1457.628,
    ];

    // Certified values from NIST
    let certified_params = Array1::from_vec(vec![
        1.2881396800E+03,
        1.4910792535E+03,
        5.8323836877E+02,
        7.5416644291E+01,
        9.6629502864E-01,
        3.9797285797E-01,
        4.9727297349E-02,
    ]);

    let certified_std_devs = Array1::from_vec(vec![
        4.6647963344E+00,
        3.9571156086E+01,
        2.8698696102E+01,
        5.5675370270E+00,
        3.1333340687E-02,
        1.4984928198E-02,
        6.5842344623E-03,
    ]);

    // Convert data to ndarray
    let x = Array2::from_shape_vec((x_data.len(), 1), x_data)?;
    let y = Array1::from_vec(y_data);

    // Create model function for Thurber
    let model_fn = |params: &Array1<f64>, x: &Array2<f64>| -> Array1<f64> {
        let b1 = params[0];
        let b2 = params[1];
        let b3 = params[2];
        let b4 = params[3];
        let b5 = params[4];
        let b6 = params[5];
        let b7 = params[6];

        x.column(0).mapv(|x_val| {
            let x2 = x_val * x_val;
            let x3 = x2 * x_val;
            let numerator = b1 + b2 * x_val + b3 * x2 + b4 * x3;
            let denominator = 1.0 + b5 * x_val + b6 * x2 + b7 * x3;
            numerator / denominator
        })
    };

    // For this complex model, we'll rely on numerical jacobian
    let model = NISTModel::new(x, y, model_fn, None);

    Ok((model, certified_params, certified_std_devs))
}

/// Test the Levenberg-Marquardt algorithm on the Thurber dataset (more challenging)
#[test]
fn test_thurber_model() {
    let (model, certified_params, _certified_std_devs) = load_thurber().unwrap();

    // Starting values (NIST suggests two starting points, we use the first)
    let initial_params = Array1::from_vec(vec![1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03]);

    // Configure and run the LM algorithm
    let config = LmConfig {
        max_iterations: 2000,
        xtol: 1e-7,
        ftol: 1e-7,
        gtol: 1e-7,
        ..LmConfig::default()
    };

    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, initial_params.clone());

    // Check if the result is successful
    if let Ok(solution) = result {
        // Extract the optimized parameters
        let optimized_params = solution.params;
        let residual_norm = solution.cost;

        // Check that the parameters are close to the certified values
        for i in 0..certified_params.len() {
            let rel_error = (optimized_params[i] - certified_params[i]).abs() / certified_params[i];
            assert!(
                rel_error < 0.05,
                "Parameter {} is not close enough to certified value. Got {}, expected {}",
                i,
                optimized_params[i],
                certified_params[i]
            );
        }

        // Check that the residual norm is reasonable
        let expected_residual_norm = 5.6427082397E+03; // From NIST certification
        let rel_error = (residual_norm - expected_residual_norm).abs() / expected_residual_norm;
        assert!(
            rel_error < 0.05,
            "Residual norm is not close enough to certified value. Got {}, expected {}",
            residual_norm,
            expected_residual_norm
        );

        println!("Thurber test passed! Parameters: {:?}", optimized_params);
    } else {
        panic!("LM optimization failed: {:?}", result.err());
    }
}
