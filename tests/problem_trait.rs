//! Integration tests for the Problem trait.

use approx::assert_relative_eq;
use lmopt_rs::{LmOptError, Problem, Result};
use ndarray::{array, Array1, Array2};

/// A simple linear model for testing: f(x) = a * x + b
struct LinearModel {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl LinearModel {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        assert_eq!(
            x_data.len(),
            y_data.len(),
            "x and y data must have the same length"
        );
        Self { x_data, y_data }
    }

    /// Create a test problem with known solution: y = 2x + 3
    fn create_test_problem() -> Self {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 7.0, 9.0, 11.0, 13.0]; // Exactly 2x + 3
        Self::new(x, y)
    }

    /// Create a test problem with noise: y = 2x + 3 + noise
    fn create_noisy_problem() -> Self {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.1, 6.9, 9.2, 10.8, 13.1]; // Approximately 2x + 3
        Self::new(x, y)
    }
}

impl Problem for LinearModel {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 2 {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        let a = params[0];
        let b = params[1];

        let residuals = self
            .x_data
            .iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| a * x + b - y)
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(residuals))
    }

    fn parameter_count(&self) -> usize {
        2 // a and b
    }

    fn residual_count(&self) -> usize {
        self.x_data.len()
    }

    // Custom Jacobian implementation for the linear model
    fn jacobian(&self, _params: &Array1<f64>) -> Result<Array2<f64>> {
        let n = self.x_data.len();
        let mut jac = Array2::zeros((n, 2));

        for i in 0..n {
            // Derivative with respect to a
            jac[[i, 0]] = self.x_data[i];
            // Derivative with respect to b
            jac[[i, 1]] = 1.0;
        }

        Ok(jac)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

/// Exponential model: f(x) = a * exp(-b * x)
struct ExponentialModel {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl ExponentialModel {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        assert_eq!(
            x_data.len(),
            y_data.len(),
            "x and y data must have the same length"
        );
        Self { x_data, y_data }
    }

    /// Create a test problem with known solution: y = 2 * exp(-0.5 * x)
    fn create_test_problem() -> Self {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![
            2.0,
            2.0 * (-0.5_f64).exp(),
            2.0 * (-1.0_f64).exp(),
            2.0 * (-1.5_f64).exp(),
            2.0 * (-2.0_f64).exp(),
        ];
        Self::new(x, y)
    }
}

impl Problem for ExponentialModel {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 2 {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        let a = params[0];
        let b = params[1];

        let residuals = self
            .x_data
            .iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| a * (-b * x).exp() - y)
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(residuals))
    }

    fn parameter_count(&self) -> usize {
        2 // a and b
    }

    fn residual_count(&self) -> usize {
        self.x_data.len()
    }

    // Using the default jacobian implementation (finite differences)
}

#[test]
fn test_linear_model_residuals() {
    let model = LinearModel::create_test_problem();

    // With exact parameters [2, 3], residuals should be all zeros
    let exact_params = array![2.0, 3.0];
    let residuals = model.eval(&exact_params).unwrap();

    // Check dimensions
    assert_eq!(residuals.len(), model.residual_count());

    // Check residuals are all close to zero
    for r in residuals.iter() {
        assert_relative_eq!(*r, 0.0, epsilon = 1e-10);
    }

    // With parameters [1, 1], residuals should be known values
    let inexact_params = array![1.0, 1.0];
    let residuals = model.eval(&inexact_params).unwrap();

    // Check residuals against expected values
    let expected = array![-3.0, -4.0, -5.0, -6.0, -7.0];
    for i in 0..residuals.len() {
        assert_relative_eq!(residuals[i], expected[i], epsilon = 1e-10);
    }
}

#[test]
fn test_linear_model_cost() {
    let model = LinearModel::create_test_problem();

    // With exact parameters [2, 3], cost should be zero
    let exact_params = array![2.0, 3.0];
    let cost = model.eval_cost(&exact_params).unwrap();
    assert_relative_eq!(cost, 0.0, epsilon = 1e-10);

    // With inexact parameters, cost should be positive
    let inexact_params = array![1.0, 1.0];
    let cost = model.eval_cost(&inexact_params).unwrap();
    assert!(cost > 0.0);

    // Calculate expected cost manually
    let residuals = model.eval(&inexact_params).unwrap();
    let expected_cost = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    assert_relative_eq!(cost, expected_cost, epsilon = 1e-10);
}

#[test]
fn test_linear_model_jacobian() {
    let model = LinearModel::create_test_problem();

    // Get Jacobian at any parameter point (Jacobian is independent of parameters for linear model)
    let params = array![1.0, 1.0];
    let jacobian = model.jacobian(&params).unwrap();

    // Check dimensions
    assert_eq!(
        jacobian.shape(),
        &[model.residual_count(), model.parameter_count()]
    );

    // Check Jacobian values - for linear model, J[i,0] = x[i], J[i,1] = 1.0
    for i in 0..model.residual_count() {
        assert_relative_eq!(jacobian[[i, 0]], model.x_data[i], epsilon = 1e-10);
        assert_relative_eq!(jacobian[[i, 1]], 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_parameter_mismatch() {
    let model = LinearModel::create_test_problem();

    // Try with too few parameters
    let too_few = array![1.0];
    let result = model.eval(&too_few);
    assert!(result.is_err());

    // Try with too many parameters
    let too_many = array![1.0, 2.0, 3.0];
    let result = model.eval(&too_many);
    assert!(result.is_err());
}

#[test]
fn test_exponential_model() {
    let model = ExponentialModel::create_test_problem();

    // With exact parameters [2, 0.5], residuals should be all zeros
    let exact_params = array![2.0, 0.5];
    let residuals = model.eval(&exact_params).unwrap();

    // Check dimensions
    assert_eq!(residuals.len(), model.residual_count());

    // Check residuals are all close to zero
    for r in residuals.iter() {
        assert_relative_eq!(*r, 0.0, epsilon = 1e-10);
    }

    // Test eval_cost for exact parameters
    let cost = model.eval_cost(&exact_params).unwrap();
    assert_relative_eq!(cost, 0.0, epsilon = 1e-10);

    // Test has_custom_jacobian
    assert!(!model.has_custom_jacobian());
}

#[test]
fn test_noisy_data() {
    let model = LinearModel::create_noisy_problem();

    // Even with the true parameters [2, 3], residuals should not be exactly zero
    let true_params = array![2.0, 3.0];
    let residuals = model.eval(&true_params).unwrap();

    // Check that residuals are small but non-zero
    let residual_norm: f64 = residuals.iter().map(|r| r.powi(2)).sum();
    assert!(residual_norm > 0.0);
    assert!(residual_norm < 0.1); // Small but non-zero
}

#[test]
fn test_large_problem() {
    // Create a large problem with many data points
    let n = 1000;
    let mut x_data = Vec::with_capacity(n);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let x = i as f64 / 100.0;
        x_data.push(x);
        y_data.push(2.0 * x + 3.0); // Exact function: y = 2x + 3
    }

    let model = LinearModel::new(Array1::from_vec(x_data), Array1::from_vec(y_data));

    // Check that parameter_count() and residual_count() return the correct values
    assert_eq!(model.parameter_count(), 2);
    assert_eq!(model.residual_count(), n);

    // With exact parameters [2, 3], residuals should be all zeros
    let exact_params = array![2.0, 3.0];
    let residuals = model.eval(&exact_params).unwrap();

    // Check dimensions
    assert_eq!(residuals.len(), n);

    // Check residuals are all close to zero
    for r in residuals.iter() {
        assert_relative_eq!(*r, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_zero_residuals() {
    // Create a problem with no residuals (empty data)
    let model = LinearModel::new(Array1::<f64>::zeros(0), Array1::<f64>::zeros(0));

    // Check that parameter_count() and residual_count() return the correct values
    assert_eq!(model.parameter_count(), 2);
    assert_eq!(model.residual_count(), 0);

    // Evaluation should work but return empty array
    let params = array![2.0, 3.0];
    let residuals = model.eval(&params).unwrap();
    assert_eq!(residuals.len(), 0);

    // Cost should be zero
    let cost = model.eval_cost(&params).unwrap();
    assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
}
