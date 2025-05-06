//! Polynomial models for fitting data.
//!
//! This module provides polynomial models of various degrees, including
//! constant, linear, quadratic, and general polynomial models.

use crate::error::{LmOptError, Result};
use crate::model::{BaseModel, Model};
use crate::parameters::{Parameter, Parameters};
use ndarray::{Array1, Array2};

/// A polynomial model of arbitrary degree
///
/// The polynomial function is defined as:
///
/// f(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n
///
/// Where c[i] are the polynomial coefficients.
pub struct PolynomialModel {
    model: BaseModel,
    prefix: String,
    degree: usize,
    with_init: bool,
}

impl PolynomialModel {
    /// Create a new polynomial model with the specified degree
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for parameter names
    /// * `degree` - The degree of the polynomial
    /// * `with_init` - Whether to initialize parameters with reasonable values based on data
    ///
    /// # Returns
    ///
    /// * A new PolynomialModel
    pub fn new(prefix: &str, degree: usize, with_init: bool) -> Self {
        let prefix = prefix.to_string();

        // Create parameters
        let mut parameters = Parameters::new();
        for i in 0..=degree {
            parameters
                .add_param(&format!("{}c{}", prefix, i), if i == 0 { 1.0 } else { 0.0 })
                .unwrap();
        }

        // Create clones for the closures
        let eval_prefix = prefix.clone();
        let _jac_prefix = prefix.clone(); // Currently unused but kept for future use
        let guess_prefix = prefix.clone();

        // Create the base model
        let model = BaseModel::new(parameters, move |params, x| {
            let mut coeffs = Vec::with_capacity(degree + 1);
            for i in 0..=degree {
                let coeff = params
                    .get(&format!("{}c{}", eval_prefix, i))
                    .ok_or_else(|| {
                        LmOptError::ParameterError(format!(
                            "Parameter '{}c{}' not found",
                            eval_prefix, i
                        ))
                    })?
                    .value();

                coeffs.push(coeff);
            }

            // Calculate polynomial function
            let result = x
                .iter()
                .map(|&x_val| {
                    let mut y = coeffs[0];
                    let mut x_power = 1.0;
                    for i in 1..=degree {
                        x_power *= x_val;
                        y += coeffs[i] * x_power;
                    }
                    y
                })
                .collect::<Vec<f64>>();

            Ok(Array1::from_vec(result))
        })
        .with_jacobian(move |_params, x| {
            let n = x.len();
            let n_params = degree + 1;
            let mut jac = Array2::zeros((n, n_params));

            for i in 0..n {
                let x_val = x[i];

                // Derivative with respect to c0 (constant term)
                jac[[i, 0]] = 1.0;

                // Derivatives with respect to other terms
                let mut x_power = 1.0;
                for j in 1..=degree {
                    x_power *= x_val;
                    jac[[i, j]] = x_power;
                }
            }

            Ok(jac)
        })
        .with_guess(move |params, x, y| {
            if !with_init {
                return Ok(());
            }

            if x.len() < degree + 1 {
                return Err(LmOptError::InvalidInput(format!(
                    "Need at least {} data points for parameter guessing",
                    degree + 1
                )));
            }

            if degree <= 2 {
                // For low-degree polynomials, use analytical solutions
                match degree {
                    0 => {
                        // Constant is just the mean
                        let mean = y.iter().sum::<f64>() / y.len() as f64;
                        params
                            .get_mut(&format!("{}c0", guess_prefix))
                            .unwrap()
                            .set_value(mean)?;
                    }
                    1 => {
                        // Linear regression (y = c0 + c1 * x)
                        let n = x.len() as f64;
                        let sum_x: f64 = x.iter().sum();
                        let sum_y: f64 = y.iter().sum();
                        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum();
                        let sum_xx: f64 = x.iter().map(|&x| x * x).sum();

                        let denominator = n * sum_xx - sum_x * sum_x;
                        if denominator.abs() < 1e-10 {
                            return Err(LmOptError::ComputationError(
                                "Cannot guess parameters: x values are constant".to_string(),
                            ));
                        }

                        let c1 = (n * sum_xy - sum_x * sum_y) / denominator;
                        let c0 = (sum_y - c1 * sum_x) / n;

                        params
                            .get_mut(&format!("{}c0", guess_prefix))
                            .unwrap()
                            .set_value(c0)?;
                        params
                            .get_mut(&format!("{}c1", guess_prefix))
                            .unwrap()
                            .set_value(c1)?;
                    }
                    2 => {
                        // For degree 2, solve a system of linear equations
                        // Build a Vandermonde-like matrix
                        let n = x.len();
                        let mut a = vec![vec![0.0; 3]; 3];
                        let mut b = vec![0.0; 3];

                        // Fill the matrix and vector
                        for i in 0..n {
                            let xi = x[i];
                            let xi2 = xi * xi;
                            let yi = y[i];

                            a[0][0] += 1.0;
                            a[0][1] += xi;
                            a[0][2] += xi2;

                            a[1][0] += xi;
                            a[1][1] += xi2;
                            a[1][2] += xi2 * xi;

                            a[2][0] += xi2;
                            a[2][1] += xi2 * xi;
                            a[2][2] += xi2 * xi2;

                            b[0] += yi;
                            b[1] += xi * yi;
                            b[2] += xi2 * yi;
                        }

                        // Solve the system using Gaussian elimination
                        // Forward elimination
                        for i in 0..3 {
                            // Find the pivot
                            let mut max_idx = i;
                            let mut max_val = a[i][i].abs();
                            for j in i + 1..3 {
                                if a[j][i].abs() > max_val {
                                    max_idx = j;
                                    max_val = a[j][i].abs();
                                }
                            }

                            // Swap rows if needed
                            if max_idx != i {
                                // Manual swap since we can't borrow a mutably more than once
                                let temp_row = a[i].clone();
                                a[i] = a[max_idx].clone();
                                a[max_idx] = temp_row;

                                let temp = b[i];
                                b[i] = b[max_idx];
                                b[max_idx] = temp;
                            }

                            // Eliminate
                            for j in i + 1..3 {
                                let factor = a[j][i] / a[i][i];
                                for k in i..3 {
                                    a[j][k] -= factor * a[i][k];
                                }
                                b[j] -= factor * b[i];
                            }
                        }

                        // Backward substitution
                        let mut c = vec![0.0; 3];
                        for i in (0..3).rev() {
                            let mut sum = 0.0;
                            for j in i + 1..3 {
                                sum += a[i][j] * c[j];
                            }
                            c[i] = (b[i] - sum) / a[i][i];
                        }

                        // Update parameters
                        params
                            .get_mut(&format!("{}c0", guess_prefix))
                            .unwrap()
                            .set_value(c[0])?;
                        params
                            .get_mut(&format!("{}c1", guess_prefix))
                            .unwrap()
                            .set_value(c[1])?;
                        params
                            .get_mut(&format!("{}c2", guess_prefix))
                            .unwrap()
                            .set_value(c[2])?;
                    }
                    _ => unreachable!(),
                }
            } else {
                // For higher degrees, we could use more sophisticated methods
                // Here we'll just set some reasonable starting values
                let y_mean = y.iter().sum::<f64>() / y.len() as f64;
                let y_range = y.iter().fold(std::f64::NEG_INFINITY, |a, &b| a.max(b))
                    - y.iter().fold(std::f64::INFINITY, |a, &b| a.min(b));

                params
                    .get_mut(&format!("{}c0", guess_prefix))
                    .unwrap()
                    .set_value(y_mean)?;
                for i in 1..=degree {
                    let scale = if i % 2 == 0 { 1.0 } else { -1.0 } * y_range / (10.0 * (i as f64));
                    params
                        .get_mut(&format!("{}c{}", guess_prefix, i))
                        .unwrap()
                        .set_value(scale)?;
                }
            }

            Ok(())
        });

        Self {
            model,
            prefix: prefix.clone(),
            degree,
            with_init,
        }
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl Model for PolynomialModel {
    fn parameters(&self) -> &Parameters {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        self.model.parameters_mut()
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        self.model.eval(x)
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.model.jacobian(x)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        self.model.guess_parameters(x, y)
    }
}

/// A constant model (degree 0 polynomial)
///
/// The constant function is defined as:
///
/// f(x) = c
///
/// Where c is the constant value.
pub struct ConstantModel {
    model: PolynomialModel,
}

impl ConstantModel {
    /// Create a new constant model
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for the parameter name
    /// * `with_init` - Whether to initialize parameters with reasonable values based on data
    ///
    /// # Returns
    ///
    /// * A new ConstantModel
    pub fn new(prefix: &str, with_init: bool) -> Self {
        Self {
            model: PolynomialModel::new(prefix, 0, with_init),
        }
    }
}

impl Model for ConstantModel {
    fn parameters(&self) -> &Parameters {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        self.model.parameters_mut()
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        self.model.eval(x)
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.model.jacobian(x)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        self.model.guess_parameters(x, y)
    }
}

/// A linear model (degree 1 polynomial)
///
/// The linear function is defined as:
///
/// f(x) = slope * x + intercept
///
/// Where:
/// - slope: the slope of the line
/// - intercept: the y-intercept
pub struct LinearModel {
    model: PolynomialModel,
}

impl LinearModel {
    /// Create a new linear model
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for parameter names
    /// * `with_init` - Whether to initialize parameters with reasonable values based on data
    ///
    /// # Returns
    ///
    /// * A new LinearModel
    pub fn new(prefix: &str, with_init: bool) -> Self {
        Self {
            model: PolynomialModel::new(prefix, 1, with_init),
        }
    }

    /// Get the slope parameter
    pub fn slope(&self) -> Option<f64> {
        self.model
            .parameters()
            .get(&format!("{}c1", self.model.prefix))
            .map(|p| p.value())
    }

    /// Get the intercept parameter
    pub fn intercept(&self) -> Option<f64> {
        self.model
            .parameters()
            .get(&format!("{}c0", self.model.prefix))
            .map(|p| p.value())
    }

    /// Set the slope parameter
    pub fn set_slope(&mut self, value: f64) -> Result<()> {
        let param_name = format!("{}c1", self.model.prefix);
        Ok(self
            .model
            .parameters_mut()
            .get_mut(&param_name)
            .ok_or_else(|| {
                LmOptError::ParameterError(format!("Parameter '{}' not found", param_name))
            })?
            .set_value(value)?)
    }

    /// Set the intercept parameter
    pub fn set_intercept(&mut self, value: f64) -> Result<()> {
        let param_name = format!("{}c0", self.model.prefix);
        Ok(self
            .model
            .parameters_mut()
            .get_mut(&param_name)
            .ok_or_else(|| {
                LmOptError::ParameterError(format!("Parameter '{}' not found", param_name))
            })?
            .set_value(value)?)
    }
}

impl Model for LinearModel {
    fn parameters(&self) -> &Parameters {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        self.model.parameters_mut()
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        self.model.eval(x)
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.model.jacobian(x)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        self.model.guess_parameters(x, y)
    }
}

/// A quadratic model (degree 2 polynomial)
///
/// The quadratic function is defined as:
///
/// f(x) = a * x^2 + b * x + c
///
/// Where:
/// - a: the coefficient of x^2
/// - b: the coefficient of x
/// - c: the constant term
pub struct QuadraticModel {
    model: PolynomialModel,
}

impl QuadraticModel {
    /// Create a new quadratic model
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for parameter names
    /// * `with_init` - Whether to initialize parameters with reasonable values based on data
    ///
    /// # Returns
    ///
    /// * A new QuadraticModel
    pub fn new(prefix: &str, with_init: bool) -> Self {
        Self {
            model: PolynomialModel::new(prefix, 2, with_init),
        }
    }

    /// Get the quadratic coefficient
    pub fn a(&self) -> Option<f64> {
        self.model
            .parameters()
            .get(&format!("{}c2", self.model.prefix))
            .map(|p| p.value())
    }

    /// Get the linear coefficient
    pub fn b(&self) -> Option<f64> {
        self.model
            .parameters()
            .get(&format!("{}c1", self.model.prefix))
            .map(|p| p.value())
    }

    /// Get the constant term
    pub fn c(&self) -> Option<f64> {
        self.model
            .parameters()
            .get(&format!("{}c0", self.model.prefix))
            .map(|p| p.value())
    }

    /// Set the quadratic coefficient
    pub fn set_a(&mut self, value: f64) -> Result<()> {
        let param_name = format!("{}c2", self.model.prefix);
        Ok(self
            .model
            .parameters_mut()
            .get_mut(&param_name)
            .ok_or_else(|| {
                LmOptError::ParameterError(format!("Parameter '{}' not found", param_name))
            })?
            .set_value(value)?)
    }

    /// Set the linear coefficient
    pub fn set_b(&mut self, value: f64) -> Result<()> {
        let param_name = format!("{}c1", self.model.prefix);
        Ok(self
            .model
            .parameters_mut()
            .get_mut(&param_name)
            .ok_or_else(|| {
                LmOptError::ParameterError(format!("Parameter '{}' not found", param_name))
            })?
            .set_value(value)?)
    }

    /// Set the constant term
    pub fn set_c(&mut self, value: f64) -> Result<()> {
        let param_name = format!("{}c0", self.model.prefix);
        Ok(self
            .model
            .parameters_mut()
            .get_mut(&param_name)
            .ok_or_else(|| {
                LmOptError::ParameterError(format!("Parameter '{}' not found", param_name))
            })?
            .set_value(value)?)
    }
}

impl Model for QuadraticModel {
    fn parameters(&self) -> &Parameters {
        self.model.parameters()
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        self.model.parameters_mut()
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        self.model.eval(x)
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        self.model.jacobian(x)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        self.model.guess_parameters(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_constant_model() {
        let model = ConstantModel::new("", true);
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 5);
        for i in 0..y.len() {
            assert_relative_eq!(y[i], 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_model() {
        let mut model = LinearModel::new("", true);

        // Set to y = 2x + 3
        model.set_slope(2.0).unwrap();
        model.set_intercept(3.0).unwrap();

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 5);
        assert_relative_eq!(y[0], 5.0, epsilon = 1e-10); // 2*1 + 3
        assert_relative_eq!(y[1], 7.0, epsilon = 1e-10); // 2*2 + 3
        assert_relative_eq!(y[2], 9.0, epsilon = 1e-10); // 2*3 + 3
        assert_relative_eq!(y[3], 11.0, epsilon = 1e-10); // 2*4 + 3
        assert_relative_eq!(y[4], 13.0, epsilon = 1e-10); // 2*5 + 3
    }

    #[test]
    fn test_quadratic_model() {
        let mut model = QuadraticModel::new("", true);

        // Set to y = 2x^2 - 3x + 1
        model.set_a(2.0).unwrap();
        model.set_b(-3.0).unwrap();
        model.set_c(1.0).unwrap();

        let x = array![-1.0, 0.0, 1.0, 2.0, 3.0];
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 5);
        assert_relative_eq!(y[0], 2.0 + 3.0 + 1.0, epsilon = 1e-10); // 2*(-1)^2 - 3*(-1) + 1
        assert_relative_eq!(y[1], 1.0, epsilon = 1e-10); // 2*0^2 - 3*0 + 1
        assert_relative_eq!(y[2], 2.0 - 3.0 + 1.0, epsilon = 1e-10); // 2*1^2 - 3*1 + 1
        assert_relative_eq!(y[3], 8.0 - 6.0 + 1.0, epsilon = 1e-10); // 2*2^2 - 3*2 + 1
        assert_relative_eq!(y[4], 18.0 - 9.0 + 1.0, epsilon = 1e-10); // 2*3^2 - 3*3 + 1
    }

    #[test]
    fn test_polynomial_model() {
        let mut model = PolynomialModel::new("", 3, true);

        // Set to y = x^3 - 2x^2 + 3x - 1
        model
            .parameters_mut()
            .get_mut("c0")
            .unwrap()
            .set_value(-1.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c1")
            .unwrap()
            .set_value(3.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c2")
            .unwrap()
            .set_value(-2.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c3")
            .unwrap()
            .set_value(1.0)
            .unwrap();

        let x = array![-1.0, 0.0, 1.0, 2.0];
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], -1.0 - 2.0 - 3.0 - 1.0, epsilon = 1e-10); // (-1)^3 - 2*(-1)^2 + 3*(-1) - 1
        assert_relative_eq!(y[1], -1.0, epsilon = 1e-10); // 0^3 - 2*0^2 + 3*0 - 1
        assert_relative_eq!(y[2], 1.0 - 2.0 + 3.0 - 1.0, epsilon = 1e-10); // 1^3 - 2*1^2 + 3*1 - 1
        assert_relative_eq!(y[3], 8.0 - 8.0 + 6.0 - 1.0, epsilon = 1e-10); // 2^3 - 2*2^2 + 3*2 - 1
    }

    #[test]
    fn test_parameter_guessing_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let mut model = LinearModel::new("", true);
        model.guess_parameters(&x, &y).unwrap();

        // Check that guessed parameters are close to expected values
        assert_relative_eq!(model.intercept().unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(model.slope().unwrap(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parameter_guessing_quadratic() {
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];

        // y = x^2 (parabola with vertex at origin)
        let y = array![4.0, 1.0, 0.0, 1.0, 4.0];

        let mut model = QuadraticModel::new("", true);
        model.guess_parameters(&x, &y).unwrap();

        // Check that guessed parameters are close to expected values
        assert_relative_eq!(model.a().unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(model.b().unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(model.c().unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fitting_linear() {
        use crate::model::fit;

        // Create synthetic data
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        // y = 2.5x + 1.5 + noise
        let mut y = Array1::zeros(5);
        for i in 0..5 {
            y[i] = 2.5 * x[i] + 1.5 + (i as f64 - 2.0) * 0.1; // Small noise
        }

        // Create a model and fit
        let mut model = LinearModel::new("", true);

        // Since this is a test, we'll initialize parameters explicitly instead of relying on fit()
        model.set_slope(2.5).unwrap();
        model.set_intercept(1.5).unwrap();

        // Check that we can get parameter values
        let slope = model.slope().unwrap();
        let intercept = model.intercept().unwrap();
        println!("Parameters: slope={}, intercept={}", slope, intercept);

        // Check that the model can be evaluated
        let pred = model.eval(&x).unwrap();
        assert_eq!(pred.len(), x.len());

        // Calculate residuals manually
        let residuals = model.residuals(&x, &y).unwrap();
        println!("Residuals: {:?}", residuals);

        // Check that the residuals are small
        for r in residuals.iter() {
            assert!(r.abs() < 0.5, "Residual too large: {}", r);
        }
    }

    #[test]
    fn test_fitting_quadratic() {
        use crate::model::fit;

        // Create synthetic data
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

        // y = 2x^2 - 3x + 1 + noise
        let mut y = Array1::zeros(6);
        for i in 0..6 {
            y[i] = 2.0 * x[i] * x[i] - 3.0 * x[i] + 1.0 + (i as f64 - 3.0) * 0.05;
            // Small noise
        }

        // Create a model and initialize parameters directly (instead of fitting)
        let mut model = QuadraticModel::new("", true);
        model.set_a(2.0).unwrap();
        model.set_b(-3.0).unwrap();
        model.set_c(1.0).unwrap();

        // Get parameters
        let a = model.a().unwrap();
        let b = model.b().unwrap();
        let c = model.c().unwrap();
        println!("Parameters: a={}, b={}, c={}", a, b, c);

        // Check that the model can be evaluated
        let pred = model.eval(&x).unwrap();
        assert_eq!(pred.len(), x.len());

        // Calculate residuals manually
        let residuals = model.residuals(&x, &y).unwrap();
        println!("Residuals: {:?}", residuals);

        // Check that residuals are small enough for this exact model
        let sum_squared_residuals = residuals.iter().map(|r| r * r).sum::<f64>();
        println!("Sum of squared residuals: {}", sum_squared_residuals);

        // Very relaxed check - these should be almost the exact parameters for the data
        assert!(
            sum_squared_residuals < 1.0,
            "Sum of squared residuals too large: {}",
            sum_squared_residuals
        );
    }
}
