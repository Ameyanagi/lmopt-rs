//! Built-in step model functions.
//!
//! This module provides step model functions such as sigmoid, heaviside, etc.

use crate::error::{LmOptError, Result};
use crate::model::{BaseModel, Model};
use crate::parameters::{Parameter, Parameters};
use ndarray::{Array1, Array2};
use std::f64::{INFINITY, NEG_INFINITY};

/// A step model using the sigmoid function.
///
/// The sigmoid function is defined as:
///
/// $f(x) = \frac{amplitude}{1 + e^{-(x - center) / sigma}} + baseline$
///
/// Parameters:
///
/// * `amplitude` - The height of the step
/// * `center` - The center of the step (x value at half height)
/// * `sigma` - The width of the transition region
/// * `baseline` - The baseline level (optional)
#[derive(Clone)]
pub struct SigmoidModel {
    /// Parameters for the model
    params: Parameters,

    /// Prefix for parameter names
    prefix: String,

    /// Base model implementation
    model: BaseModel<impl Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>> + Clone>,

    /// Whether the model has a baseline parameter
    has_baseline: bool,
}

impl Model for SigmoidModel {
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
        // Get parameter values
        let amplitude = self
            .params
            .get(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?
            .value();
        let center = self
            .params
            .get(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?
            .value();
        let sigma = self
            .params
            .get(&format!("{}sigma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?
            .value();

        let n = x.len();
        let n_params = if self.has_baseline { 4 } else { 3 };
        let mut jac = Array2::zeros((n, n_params));

        for i in 0..n {
            let x_val = x[i];
            let exp_term = f64::exp(-(x_val - center) / sigma);
            let denom = 1.0 + exp_term;
            let sigmoid = 1.0 / denom;

            // Derivative with respect to amplitude
            jac[[i, 0]] = sigmoid;

            // Derivative with respect to center
            jac[[i, 1]] = amplitude * sigmoid * sigmoid * exp_term / sigma;

            // Derivative with respect to sigma
            jac[[i, 2]] =
                amplitude * sigmoid * sigmoid * exp_term * (x_val - center) / (sigma * sigma);

            // Derivative with respect to baseline (if applicable)
            if self.has_baseline {
                jac[[i, 3]] = 1.0;
            }
        }

        Ok(jac)
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if x.len() < 3 {
            return Err(LmOptError::InvalidInput(
                "Need at least 3 data points for parameter guessing".to_string(),
            ));
        }

        // Similar to step function, but we need to estimate sigma too
        // First, sort points by x
        let mut xy: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

        xy.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find approximate baseline and amplitude from min/max values
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let amplitude = y_max - y_min;
        let baseline = y_min;

        // Estimate center as the x value where y is halfway between baseline and max
        let half_height = baseline + amplitude / 2.0;
        let mut center = (xy.first().unwrap().0 + xy.last().unwrap().0) / 2.0; // Default to midpoint

        // Find the point closest to half-height
        for i in 0..(xy.len() - 1) {
            let (x1, y1) = xy[i];
            let (x2, y2) = xy[i + 1];

            if (y1 < half_height && y2 >= half_height) || (y1 >= half_height && y2 < half_height) {
                // Linear interpolation to find better estimate of center
                center = x1 + (half_height - y1) * (x2 - x1) / (y2 - y1);
                break;
            }
        }

        // Estimate sigma by finding width around inflection point
        // For sigmoid, this is approximately the width for 25% to 75% height
        let quarter_height = baseline + amplitude * 0.25;
        let three_quarter_height = baseline + amplitude * 0.75;

        let mut x_low = xy.first().unwrap().0;
        let mut x_high = xy.last().unwrap().0;

        for i in 0..(xy.len() - 1) {
            let (x1, y1) = xy[i];
            let (x2, y2) = xy[i + 1];

            if (y1 < quarter_height && y2 >= quarter_height)
                || (y1 >= quarter_height && y2 < quarter_height)
            {
                // Linear interpolation
                x_low = x1 + (quarter_height - y1) * (x2 - x1) / (y2 - y1);
            }

            if (y1 < three_quarter_height && y2 >= three_quarter_height)
                || (y1 >= three_quarter_height && y2 < three_quarter_height)
            {
                // Linear interpolation
                x_high = x1 + (three_quarter_height - y1) * (x2 - x1) / (y2 - y1);
            }
        }

        // Sigma is approximately the width between these points
        let sigma = (x_high - x_low).abs();
        if sigma < 1e-10 {
            // Can't reliably determine sigma, use a default
            let sigma = (xy.last().unwrap().0 - xy.first().unwrap().0).abs() / 10.0;
            self.params
                .get_mut(&format!("{}sigma", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?
                .set_value(sigma.max(1e-5))?;
        } else {
            self.params
                .get_mut(&format!("{}sigma", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?
                .set_value(sigma)?;
        }

        // Set the parameters
        self.params
            .get_mut(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?
            .set_value(amplitude)?;

        self.params
            .get_mut(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?
            .set_value(center)?;

        if self.has_baseline {
            self.params
                .get_mut(&format!("{}baseline", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}baseline", self.prefix)))?
                .set_value(baseline)?;
        }

        Ok(())
    }
}

impl SigmoidModel {
    /// Create a new sigmoid model with the specified parameter prefix
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for parameter names
    /// * `with_baseline` - Whether to include a baseline parameter
    ///
    /// # Returns
    ///
    /// * A new SigmoidModel
    pub fn new(prefix: &str, with_baseline: bool) -> Self {
        let prefix = prefix.to_string();

        // Create parameters
        let mut parameters = Parameters::new();
        parameters
            .add_param(&format!("{}amplitude", prefix), 1.0)
            .unwrap();
        parameters
            .add_param(&format!("{}center", prefix), 0.0)
            .unwrap();
        parameters
            .add_param(&format!("{}sigma", prefix), 1.0)
            .unwrap();

        if with_baseline {
            parameters
                .add_param(&format!("{}baseline", prefix), 0.0)
                .unwrap();
        }

        // Ensure sigma is positive
        parameters
            .get_mut(&format!("{}sigma", prefix))
            .unwrap()
            .set_min(0.0)
            .unwrap();

        // Create clones for the closures
        let eval_prefix = prefix.clone();
        let has_baseline = with_baseline;

        // Create the base model
        let model = BaseModel::new(parameters, move |params, x| {
            let amplitude = params
                .get(&format!("{}amplitude", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}amplitude' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let center = params
                .get(&format!("{}center", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}center' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let sigma = params
                .get(&format!("{}sigma", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}sigma' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let baseline = if has_baseline {
                params
                    .get(&format!("{}baseline", eval_prefix))
                    .ok_or_else(|| {
                        LmOptError::ParameterError(format!(
                            "Parameter '{}baseline' not found",
                            eval_prefix
                        ))
                    })?
                    .value()
            } else {
                0.0
            };

            // Calculate sigmoid function
            let result = x
                .iter()
                .map(|&x_val| amplitude / (1.0 + f64::exp(-(x_val - center) / sigma)) + baseline)
                .collect::<Vec<f64>>();

            Ok(Array1::from_vec(result))
        });

        // Create a SigmoidModel
        SigmoidModel {
            params: model.parameters().clone(),
            prefix,
            model,
            has_baseline: with_baseline,
        }
    }
}

/// A step model using the linear step function.
///
/// The linear step function is defined as:
///
/// $f(x) = amplitude * (x - center) / (|x - center| + sigma) + baseline$
///
/// This function makes a smooth transition from -amplitude/2 to +amplitude/2
/// with the center of the step at the specified center point. The sigma parameter
/// controls the width of the transition region.
///
/// Parameters:
///
/// * `amplitude` - The height of the step
/// * `center` - The center of the step (x value at zero height)
/// * `sigma` - The width of the transition region
/// * `baseline` - The baseline level (optional)
#[derive(Clone)]
pub struct LinearStepModel {
    /// Parameters for the model
    params: Parameters,

    /// Prefix for parameter names
    prefix: String,

    /// Base model implementation
    model: BaseModel<impl Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>> + Clone>,

    /// Whether the model has a baseline parameter
    has_baseline: bool,
}

impl Model for LinearStepModel {
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
        // Get parameter values
        let amplitude = self
            .params
            .get(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?
            .value();
        let center = self
            .params
            .get(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?
            .value();
        let sigma = self
            .params
            .get(&format!("{}sigma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?
            .value();

        let n = x.len();
        let n_params = if self.has_baseline { 4 } else { 3 };
        let mut jac = Array2::zeros((n, n_params));

        for i in 0..n {
            let x_val = x[i];
            let diff = x_val - center;
            let abs_diff = diff.abs();
            let denom = abs_diff + sigma;
            let denom_squared = denom * denom;

            // Derivative with respect to amplitude
            jac[[i, 0]] = diff / denom;

            // Derivative with respect to center
            jac[[i, 1]] = -amplitude / denom + amplitude * diff * diff.signum() / denom_squared;

            // Derivative with respect to sigma
            jac[[i, 2]] = -amplitude * diff / denom_squared;

            // Derivative with respect to baseline (if applicable)
            if self.has_baseline {
                jac[[i, 3]] = 1.0;
            }
        }

        Ok(jac)
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if x.len() < 3 {
            return Err(LmOptError::InvalidInput(
                "Need at least 3 data points for parameter guessing".to_string(),
            ));
        }

        // Linear step is similar to sigmoid but has different scaling
        // First, sort points by x
        let mut xy: Vec<(f64, f64)> = x.iter().zip(y.iter()).map(|(&x, &y)| (x, y)).collect();

        xy.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find approximate baseline from average
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let amplitude = y_max - y_min;
        let center_y = (y_min + y_max) / 2.0;

        // Find center as x where y is closest to center_y
        let mut center = xy[0].0;
        let mut min_dist = f64::INFINITY;

        for (x_val, y_val) in &xy {
            let dist = (y_val - center_y).abs();
            if dist < min_dist {
                min_dist = dist;
                center = *x_val;
            }
        }

        // Estimate sigma from the rate of change
        // Linear step has max slope of amplitude/(4*sigma) at center
        // Find points on either side of center to estimate slope
        let mut left_x = xy.first().unwrap().0;
        let mut left_y = xy.first().unwrap().1;
        let mut right_x = xy.last().unwrap().0;
        let mut right_y = xy.last().unwrap().1;

        for i in 0..(xy.len() - 1) {
            if xy[i].0 <= center && xy[i + 1].0 > center {
                left_x = xy[i].0;
                left_y = xy[i].1;
                right_x = xy[i + 1].0;
                right_y = xy[i + 1].1;
                break;
            }
        }

        let slope = if (right_x - left_x).abs() > 1e-10 {
            (right_y - left_y) / (right_x - left_x)
        } else {
            // Default if can't determine slope
            amplitude / (2.0 * (xy.last().unwrap().0 - xy.first().unwrap().0).abs())
        };

        let sigma = if slope.abs() > 1e-10 {
            amplitude / (4.0 * slope.abs())
        } else {
            // Default if slope is near zero
            (xy.last().unwrap().0 - xy.first().unwrap().0).abs() / 10.0
        };

        // Set the parameters
        self.params
            .get_mut(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?
            .set_value(amplitude)?;

        self.params
            .get_mut(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?
            .set_value(center)?;

        self.params
            .get_mut(&format!("{}sigma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?
            .set_value(sigma.max(1e-5))?;

        if self.has_baseline {
            let baseline = center_y - amplitude / 2.0;
            self.params
                .get_mut(&format!("{}baseline", self.prefix))
                .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}baseline", self.prefix)))?
                .set_value(baseline)?;
        }

        Ok(())
    }
}

impl LinearStepModel {
    /// Create a new linear step model with the specified parameter prefix
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for parameter names
    /// * `with_baseline` - Whether to include a baseline parameter
    ///
    /// # Returns
    ///
    /// * A new LinearStepModel
    pub fn new(prefix: &str, with_baseline: bool) -> Self {
        let prefix = prefix.to_string();

        // Create parameters
        let mut parameters = Parameters::new();
        parameters
            .add_param(&format!("{}amplitude", prefix), 1.0)
            .unwrap();
        parameters
            .add_param(&format!("{}center", prefix), 0.0)
            .unwrap();
        parameters
            .add_param(&format!("{}sigma", prefix), 1.0)
            .unwrap();

        if with_baseline {
            parameters
                .add_param(&format!("{}baseline", prefix), 0.0)
                .unwrap();
        }

        // Ensure sigma is positive
        parameters
            .get_mut(&format!("{}sigma", prefix))
            .unwrap()
            .set_min(0.0)
            .unwrap();

        // Create clones for the closures
        let eval_prefix = prefix.clone();
        let has_baseline = with_baseline;

        // Create the base model
        let model = BaseModel::new(parameters, move |params, x| {
            let amplitude = params
                .get(&format!("{}amplitude", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}amplitude' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let center = params
                .get(&format!("{}center", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}center' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let sigma = params
                .get(&format!("{}sigma", eval_prefix))
                .ok_or_else(|| {
                    LmOptError::ParameterError(format!(
                        "Parameter '{}sigma' not found",
                        eval_prefix
                    ))
                })?
                .value();

            let baseline = if has_baseline {
                params
                    .get(&format!("{}baseline", eval_prefix))
                    .ok_or_else(|| {
                        LmOptError::ParameterError(format!(
                            "Parameter '{}baseline' not found",
                            eval_prefix
                        ))
                    })?
                    .value()
            } else {
                0.0
            };

            // Calculate linear step function
            let result = x
                .iter()
                .map(|&x_val| {
                    let diff = x_val - center;
                    amplitude * diff / (diff.abs() + sigma) + baseline
                })
                .collect::<Vec<f64>>();

            Ok(Array1::from_vec(result))
        });

        // Create a LinearStepModel
        LinearStepModel {
            params: model.parameters().clone(),
            prefix,
            model,
            has_baseline: with_baseline,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_model_evaluation() {
        let model = SigmoidModel::new("", true);
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        let y = model.eval(&x).unwrap();

        // Expected values for sigmoid with amplitude=1, center=0, sigma=1, baseline=0
        let expected = [
            1.0 / (1.0 + f64::exp(2.0)) + 0.0,
            1.0 / (1.0 + f64::exp(1.0)) + 0.0,
            1.0 / (1.0 + f64::exp(0.0)) + 0.0,
            1.0 / (1.0 + f64::exp(-1.0)) + 0.0,
            1.0 / (1.0 + f64::exp(-2.0)) + 0.0,
        ];

        assert_eq!(y.len(), 5);
        for i in 0..5 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parameter_initialization() {
        // Create synthetic data with a sigmoid step
        let x = Array1::linspace(-10.0, 10.0, 100);
        let y: Array1<f64> = x
            .iter()
            .map(|&x_val| {
                let arg = -((x_val - 2.0) / 1.5);
                3.0 / (1.0 + f64::exp(arg)) + 0.5
            })
            .collect();

        // Create model and initialize parameters
        let mut sigmoid = SigmoidModel::new("s_", true);
        sigmoid.guess_parameters(&x, &y).unwrap();

        // Check that parameters are reasonably close to the true values
        let amp = sigmoid.parameters().get("s_amplitude").unwrap().value();
        let center = sigmoid.parameters().get("s_center").unwrap().value();
        let sigma = sigmoid.parameters().get("s_sigma").unwrap().value();
        let baseline = sigmoid.parameters().get("s_baseline").unwrap().value();

        assert_relative_eq!(amp, 3.0, epsilon = 0.5);
        assert_relative_eq!(center, 2.0, epsilon = 1.0);
        assert_relative_eq!(sigma, 1.5, epsilon = 1.0);
        assert_relative_eq!(baseline, 0.5, epsilon = 0.5);
    }

    #[test]
    fn test_linear_step_model() {
        let model = LinearStepModel::new("", false);
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        let y = model.eval(&x).unwrap();

        // Expected values for linear step with amplitude=1, center=0, sigma=1, baseline=0
        let expected = [
            1.0 * -2.0 / (2.0 + 1.0) + 0.0,
            1.0 * -1.0 / (1.0 + 1.0) + 0.0,
            0.0 + 0.0,
            1.0 * 1.0 / (1.0 + 1.0) + 0.0,
            1.0 * 2.0 / (2.0 + 1.0) + 0.0,
        ];

        assert_eq!(y.len(), 5);
        for i in 0..5 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fit_sigmoid_model() {
        use crate::model::fit;

        // Create synthetic data with a sigmoid step
        let x = Array1::linspace(-5.0, 5.0, 50);
        let y: Array1<f64> = x
            .iter()
            .map(|&x_val| {
                let arg = -((x_val - 1.0) / 0.8);
                2.0 / (1.0 + f64::exp(arg)) + 0.3
            })
            .collect();

        // Create model and fit
        let mut model = SigmoidModel::new("", true);
        let _result = fit(&mut model, x.clone(), y.clone()).unwrap();

        // Now check if the parameters are reasonable
        let amp = model.parameters().get("amplitude").unwrap().value();
        let center = model.parameters().get("center").unwrap().value();
        let sigma = model.parameters().get("sigma").unwrap().value();
        let baseline = model.parameters().get("baseline").unwrap().value();

        // Model should recover parameters accurately
        assert_relative_eq!(amp, 2.0, epsilon = 0.1);
        assert_relative_eq!(center, 1.0, epsilon = 0.1);
        assert_relative_eq!(sigma, 0.8, epsilon = 0.1);
        assert_relative_eq!(baseline, 0.3, epsilon = 0.1);
    }
}
