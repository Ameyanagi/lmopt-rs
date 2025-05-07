//! Polynomial model implementations.
//!
//! This module provides polynomial models of arbitrary degree.

use crate::error::{LmOptError, Result};
use crate::model::{BaseModel, Model};
use crate::parameters::{Parameter, Parameters};
use ndarray::{Array1, Array2};

/// A polynomial model of arbitrary degree.
///
/// The polynomial function is defined as:
///
/// $f(x) = c_0 + c_1 x + c_2 x^2 + \ldots + c_n x^n$
///
/// Parameters:
///
/// * `c0` - Constant term
/// * `c1` - Linear term
/// * `c2` - Quadratic term
/// * ... and so on
#[derive(Clone)]
pub struct PolynomialModel {
    /// Parameters for the model
    pub(crate) params: Parameters,

    /// Prefix for parameter names
    pub(crate) prefix: String,

    /// Base model implementation
    pub(crate) model: BaseModel<impl Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>> + Clone>,

    /// Degree of the polynomial
    pub(crate) degree: usize,
}

impl Model for PolynomialModel {
    fn parameters(&self) -> &Parameters {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.params
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        self.model.eval(x)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        let n = x.len();
        let n_params = self.degree + 1;
        let mut jac = Array2::zeros((n, n_params));

        for i in 0..n {
            let x_val = x[i];

            // Derivative with respect to c0 (constant term)
            jac[[i, 0]] = 1.0;

            // Derivatives with respect to other terms
            let mut x_power = 1.0;
            for j in 1..=self.degree {
                x_power *= x_val;
                jac[[i, j]] = x_power;
            }
        }

        Ok(jac)
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if x.len() < 2 {
            return Err(LmOptError::InvalidInput(
                "Need at least 2 data points for parameter guessing".to_string(),
            ));
        }

        // For degree 0, just use the mean of y
        if self.degree == 0 {
            let mean_y = y.iter().sum::<f64>() / y.len() as f64;
            self.params
                .get_mut(&format!("{}c0", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c0", self.prefix)))?
                .set_value(mean_y)?;
            return Ok(());
        }

        // For degree 1, use linear regression
        if self.degree == 1 {
            let n = x.len() as f64;
            let sum_x = x.iter().sum::<f64>();
            let sum_y = y.iter().sum::<f64>();
            let sum_xy = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>();
            let sum_x2 = x.iter().map(|&x| x * x).sum::<f64>();

            let denominator = n * sum_x2 - sum_x * sum_x;
            if denominator.abs() < 1e-10 {
                // If x values are all the same, just use mean of y as constant
                let mean_y = sum_y / n;
                self.params
                    .get_mut(&format!("{}c0", self.prefix))
                    .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c0", self.prefix)))?
                    .set_value(mean_y)?;
                self.params
                    .get_mut(&format!("{}c1", self.prefix))
                    .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c1", self.prefix)))?
                    .set_value(0.0)?;
                return Ok(());
            }

            let slope = (n * sum_xy - sum_x * sum_y) / denominator;
            let intercept = (sum_y - slope * sum_x) / n;

            self.params
                .get_mut(&format!("{}c0", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c0", self.prefix)))?
                .set_value(intercept)?;
            self.params
                .get_mut(&format!("{}c1", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c1", self.prefix)))?
                .set_value(slope)?;

            return Ok(());
        }

        // For degree 2, use quadratic regression
        if self.degree == 2 {
            // Fit quadratic model y = c0 + c1*x + c2*x^2
            // We'll use normal equations in matrix form: A*c = b
            // Where:
            // A = [ n         sum(x)     sum(x^2)    ]
            //     [ sum(x)    sum(x^2)   sum(x^3)    ]
            //     [ sum(x^2)  sum(x^3)   sum(x^4)    ]
            //
            // b = [ sum(y)      ]
            //     [ sum(x*y)    ]
            //     [ sum(x^2*y)  ]
            //
            // c = [ c0 c1 c2 ]^T

            // We'll just use arrays for simplicity
            let n = x.len();

            // Initialize matrices
            let mut a = vec![vec![0.0; 3]; 3];
            let mut b = vec![0.0; 3];

            // Compute sums
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
                let mut max_val = (a[i][i] as f64).abs();
                for j in i + 1..3 {
                    if (a[j][i] as f64).abs() > max_val {
                        max_idx = j;
                        max_val = (a[j][i] as f64).abs();
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

                // Eliminate below
                for j in i + 1..3 {
                    let factor = a[j][i] / a[i][i];
                    for k in i..3 {
                        a[j][k] -= factor * a[i][k];
                    }
                    b[j] -= factor * b[i];
                }
            }

            // Back substitution
            let mut c = vec![0.0; 3];
            for i in (0..3).rev() {
                let mut sum = 0.0;
                for j in i + 1..3 {
                    sum += a[i][j] * c[j];
                }
                c[i] = (b[i] - sum) / a[i][i];
            }

            // Set the parameters
            self.params
                .get_mut(&format!("{}c0", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c0", self.prefix)))?
                .set_value(c[0])?;
            self.params
                .get_mut(&format!("{}c1", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c1", self.prefix)))?
                .set_value(c[1])?;
            self.params
                .get_mut(&format!("{}c2", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c2", self.prefix)))?
                .set_value(c[2])?;

            return Ok(());
        }

        // For higher degrees, default to linear fit
        // This could be improved with polynomial regression algorithms
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>();
        let sum_x2 = x.iter().map(|&x| x * x).sum::<f64>();

        let denominator = n * sum_x2 - sum_x * sum_x;
        let slope = if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        };
        let intercept = (sum_y - slope * sum_x) / n;

        // Set all parameters to 0 except c0 and c1
        for i in 0..=self.degree {
            let param = self
                .params
                .get_mut(&format!("{}c{}", self.prefix, i))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}c{}", self.prefix, i)))?;

            match i {
                0 => param.set_value(intercept)?,
                1 => param.set_value(slope)?,
                _ => param.set_value(0.0)?,
            }
        }

        Ok(())
    }
}

impl PolynomialModel {
    /// Create a new polynomial model with the specified degree.
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
                .add_param(&format!("{}c{}", prefix, i), 0.0)
                .unwrap();
        }

        // Create clones for the closures
        let eval_prefix = prefix.clone();

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
        });

        // Create a PolynomialModel
        PolynomialModel {
            params: model.parameters().clone(),
            prefix,
            model,
            degree,
        }
    }
}

/// A constant model (polynomial of degree 0).
///
/// The constant function is defined as:
///
/// $f(x) = c$
///
/// Parameters:
///
/// * `c0` - Constant value
pub type ConstantModel = PolynomialModel;

/// A linear model (polynomial of degree 1).
///
/// The linear function is defined as:
///
/// $f(x) = a x + b$
///
/// Parameters:
///
/// * `c0` - Intercept (b)
/// * `c1` - Slope (a)
pub type LinearModel = PolynomialModel;

/// A quadratic model (polynomial of degree 2).
///
/// The quadratic function is defined as:
///
/// $f(x) = a x^2 + b x + c$
///
/// Parameters:
///
/// * `c0` - Constant term (c)
/// * `c1` - Linear term (b)
/// * `c2` - Quadratic term (a)
pub type QuadraticModel = PolynomialModel;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_constant_model() {
        let mut model = ConstantModel::new("", 0, false);
        model
            .parameters_mut()
            .get_mut("c0")
            .unwrap()
            .set_value(5.0)
            .unwrap();

        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 4);
        for &value in y.iter() {
            assert_relative_eq!(value, 5.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_linear_model() {
        let mut model = LinearModel::new("", 1, false);
        model
            .parameters_mut()
            .get_mut("c0")
            .unwrap()
            .set_value(2.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c1")
            .unwrap()
            .set_value(3.0)
            .unwrap();

        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], 2.0 - 3.0, epsilon = 1e-10); // -1.0
        assert_relative_eq!(y[1], 2.0, epsilon = 1e-10); // 2.0
        assert_relative_eq!(y[2], 2.0 + 3.0, epsilon = 1e-10); // 5.0
        assert_relative_eq!(y[3], 2.0 + 6.0, epsilon = 1e-10); // 8.0
    }

    #[test]
    fn test_quadratic_model() {
        let mut model = QuadraticModel::new("", 2, false);
        model
            .parameters_mut()
            .get_mut("c0")
            .unwrap()
            .set_value(1.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c1")
            .unwrap()
            .set_value(2.0)
            .unwrap();
        model
            .parameters_mut()
            .get_mut("c2")
            .unwrap()
            .set_value(3.0)
            .unwrap();

        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 4);
        assert_relative_eq!(y[0], 1.0 - 2.0 + 3.0, epsilon = 1e-10); // 2.0
        assert_relative_eq!(y[1], 1.0, epsilon = 1e-10); // 1.0
        assert_relative_eq!(y[2], 1.0 + 2.0 + 3.0, epsilon = 1e-10); // 6.0
        assert_relative_eq!(y[3], 1.0 + 4.0 + 12.0, epsilon = 1e-10); // 17.0
    }

    #[test]
    fn test_parameter_guess_constant() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![5.0, 5.0, 5.0, 5.0, 5.0]);

        let mut model = ConstantModel::new("", 0, true);
        model.guess_parameters(&x, &y).unwrap();

        let c0 = model.parameters().get("c0").unwrap().value();
        assert_relative_eq!(c0, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parameter_guess_linear() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]); // y = 2x + 1

        let mut model = LinearModel::new("", 1, true);
        model.guess_parameters(&x, &y).unwrap();

        let c0 = model.parameters().get("c0").unwrap().value();
        let c1 = model.parameters().get("c1").unwrap().value();

        assert_relative_eq!(c0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(c1, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parameter_guess_quadratic() {
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = Array1::from_vec(vec![9.0, 4.0, 1.0, 4.0, 9.0]); // y = 2x^2 + 0x + 1

        let mut model = QuadraticModel::new("", 2, true);
        model.guess_parameters(&x, &y).unwrap();

        let c0 = model.parameters().get("c0").unwrap().value();
        let c1 = model.parameters().get("c1").unwrap().value();
        let c2 = model.parameters().get("c2").unwrap().value();

        assert_relative_eq!(c0, 1.0, epsilon = 1e-8);
        assert_relative_eq!(c1, 0.0, epsilon = 1e-8);
        assert_relative_eq!(c2, 2.0, epsilon = 1e-8);
    }

    #[test]
    fn test_jacobian() {
        let model = QuadraticModel::new("", 2, false);
        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0]);

        let jac = model.jacobian(&x).unwrap();

        // Expected Jacobian:
        // [1.0, -1.0, 1.0]
        // [1.0, 0.0, 0.0]
        // [1.0, 1.0, 1.0]
        // [1.0, 2.0, 4.0]

        assert_eq!(jac.shape(), &[4, 3]);

        // Check derivatives with respect to c0 (constant)
        assert_relative_eq!(jac[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[1, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[2, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[3, 0]], 1.0, epsilon = 1e-10);

        // Check derivatives with respect to c1 (linear)
        assert_relative_eq!(jac[[0, 1]], -1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[1, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[2, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[3, 1]], 2.0, epsilon = 1e-10);

        // Check derivatives with respect to c2 (quadratic)
        assert_relative_eq!(jac[[0, 2]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[1, 2]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[2, 2]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(jac[[3, 2]], 4.0, epsilon = 1e-10);
    }
}
