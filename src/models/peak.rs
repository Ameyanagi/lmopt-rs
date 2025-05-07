//! Peak models for fitting data.
//!
//! This module provides models for common peak functions like Gaussian and Lorentzian,
//! which are widely used in spectroscopy, diffraction, and other scientific fields.

use crate::error::{LmOptError, Result};
use crate::model::{BaseModel, Model};
use crate::parameters::{Parameter, Parameters};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// A Gaussian peak model.
///
/// The Gaussian function is defined as:
/// f(x) = amplitude * exp(-(x - center)² / (2 * sigma²)) + baseline
///
/// It is characterized by:
/// - `amplitude`: The height of the peak
/// - `center`: The position of the peak center
/// - `sigma`: The standard deviation, controlling the width
/// - `baseline`: The baseline offset
///
/// The Full Width at Half Maximum (FWHM) is related to sigma by:
/// FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
#[derive(Debug, Clone)]
pub struct GaussianModel {
    params: Parameters,
    prefix: String,
    with_init: bool,
}

impl GaussianModel {
    /// Create a new Gaussian model.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Prefix for parameter names
    /// * `with_baseline` - Whether to include a baseline parameter
    ///
    /// # Returns
    ///
    /// * A new GaussianModel instance
    pub fn new(prefix: &str, with_baseline: bool) -> Self {
        let mut params = Parameters::new();

        // Add parameters with default values and bounds
        let amplitude_name = format!("{}amplitude", prefix);
        let center_name = format!("{}center", prefix);
        let sigma_name = format!("{}sigma", prefix);
        let baseline_name = format!("{}baseline", prefix);

        params
            .add_param_with_bounds(&amplitude_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();
        params.add_param(&center_name, 0.0).unwrap();
        params
            .add_param_with_bounds(&sigma_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();

        if with_baseline {
            params.add_param(&baseline_name, 0.0).unwrap();
        }

        // Add a derived parameter for FWHM based on sigma
        // FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.3548 * sigma
        let fwhm_name = format!("{}fwhm", prefix);
        let sigma_expr = format!("{}sigma", prefix);
        params
            .add_param_with_expr(&fwhm_name, 2.3548, &format!("2.3548 * {}", sigma_expr))
            .unwrap();

        Self {
            params,
            prefix: prefix.to_string(),
            with_init: true, // Default to auto-initialization
        }
    }

    /// Set whether to auto-initialize parameters based on data.
    pub fn with_initialization(mut self, init: bool) -> Self {
        self.with_init = init;
        self
    }
}

impl Model for GaussianModel {
    fn parameters(&self) -> &Parameters {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.params
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
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

        // Get baseline if it exists
        let baseline = self
            .params
            .get(&format!("{}baseline", self.prefix))
            .map(|p| p.value())
            .unwrap_or(0.0);

        // Compute Gaussian function for each x value
        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
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

        // Count the actual number of varying parameters
        let has_baseline = self
            .params
            .get(&format!("{}baseline", self.prefix))
            .is_some();
        let n_params = if has_baseline { 4 } else { 3 };

        // Initialize Jacobian matrix
        let mut jac = Array2::zeros((x.len(), n_params));

        // Compute partial derivatives for each x value and parameter
        for (i, &x_val) in x.iter().enumerate() {
            let arg = (x_val - center) / sigma;
            let exp_term = (-0.5 * arg * arg).exp();

            // d/d(amplitude)
            jac[[i, 0]] = exp_term;

            // d/d(center)
            jac[[i, 1]] = amplitude * exp_term * arg / sigma;

            // d/d(sigma)
            jac[[i, 2]] = amplitude * exp_term * arg * arg / sigma;

            // d/d(baseline) if applicable
            if has_baseline {
                jac[[i, 3]] = 1.0;
            }
        }

        Ok(jac)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if !self.with_init {
            return Ok(());
        }

        if x.len() != y.len() || x.is_empty() {
            return Err(LmOptError::DimensionMismatch(
                "x and y must have the same non-zero length".to_string(),
            ));
        }

        // Find the maximum value in y and its corresponding x
        let (max_idx, &max_y) = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| {
                LmOptError::ComputationError("Failed to find maximum y value".to_string())
            })?;

        let max_x = x[max_idx];

        // Find the minimum value for baseline estimation
        let min_y = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Estimate amplitude as the height above baseline
        let amplitude_est = max_y - min_y;

        // Estimate center as the x value at the maximum y
        let center_est = max_x;

        // Estimate sigma by checking width around center point
        // Look for points where y falls to half the peak height above baseline
        let half_height = min_y + amplitude_est / 2.0;

        // Find points to the left and right of center that cross half height
        let mut left_idx = max_idx;
        while left_idx > 0 && y[left_idx] > half_height {
            left_idx -= 1;
        }

        let mut right_idx = max_idx;
        while right_idx < y.len() - 1 && y[right_idx] > half_height {
            right_idx += 1;
        }

        // Estimate FWHM from these points
        let fwhm_est = if left_idx < right_idx {
            (x[right_idx] - x[left_idx]).abs()
        } else {
            // Default if we couldn't find crossings
            (x[x.len() - 1] - x[0]).abs() / 5.0
        };

        // Convert FWHM to sigma: sigma = FWHM / (2 * sqrt(2 * ln(2)))
        let sigma_est = fwhm_est / 2.3548;

        // Update parameters with estimates
        let amp_param = self
            .params
            .get_mut(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?;
        amp_param.set_value(amplitude_est.max(0.0))?;

        let center_param = self
            .params
            .get_mut(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?;
        center_param.set_value(center_est)?;

        let sigma_param = self
            .params
            .get_mut(&format!("{}sigma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?;
        sigma_param.set_value(sigma_est.max(0.1))?;

        // Set baseline if applicable
        if let Some(baseline_param) = self.params.get_mut(&format!("{}baseline", self.prefix)) {
            baseline_param.set_value(min_y)?;
        }

        // The derived parameter fwhm will be updated automatically when the parameters
        // are used since it uses an expression

        Ok(())
    }
}

/// A Lorentzian peak model.
///
/// The Lorentzian function is defined as:
/// f(x) = amplitude * (gamma² / ((x - center)² + gamma²)) + baseline
///
/// It is characterized by:
/// - `amplitude`: The height of the peak
/// - `center`: The position of the peak center
/// - `gamma`: The half-width at half-maximum (HWHM)
/// - `baseline`: The baseline offset
///
/// The Full Width at Half Maximum (FWHM) is 2*gamma.
#[derive(Debug, Clone)]
pub struct LorentzianModel {
    params: Parameters,
    prefix: String,
    with_init: bool,
}

impl LorentzianModel {
    /// Create a new Lorentzian model.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Prefix for parameter names
    /// * `with_baseline` - Whether to include a baseline parameter
    ///
    /// # Returns
    ///
    /// * A new LorentzianModel instance
    pub fn new(prefix: &str, with_baseline: bool) -> Self {
        let mut params = Parameters::new();

        // Add parameters with default values and bounds
        let amplitude_name = format!("{}amplitude", prefix);
        let center_name = format!("{}center", prefix);
        let gamma_name = format!("{}gamma", prefix);
        let baseline_name = format!("{}baseline", prefix);

        params
            .add_param_with_bounds(&amplitude_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();
        params.add_param(&center_name, 0.0).unwrap();
        params
            .add_param_with_bounds(&gamma_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();

        if with_baseline {
            params.add_param(&baseline_name, 0.0).unwrap();
        }

        // Add a derived parameter for FWHM = 2*gamma
        let fwhm_name = format!("{}fwhm", prefix);
        let gamma_expr = format!("{}gamma", prefix);
        params
            .add_param_with_expr(&fwhm_name, 2.0, &format!("2.0 * {}", gamma_expr))
            .unwrap();

        Self {
            params,
            prefix: prefix.to_string(),
            with_init: true, // Default to auto-initialization
        }
    }

    /// Set whether to auto-initialize parameters based on data.
    pub fn with_initialization(mut self, init: bool) -> Self {
        self.with_init = init;
        self
    }
}

impl Model for LorentzianModel {
    fn parameters(&self) -> &Parameters {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.params
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
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
        let gamma = self
            .params
            .get(&format!("{}gamma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}gamma", self.prefix)))?
            .value();

        // Get baseline if it exists
        let baseline = self
            .params
            .get(&format!("{}baseline", self.prefix))
            .map(|p| p.value())
            .unwrap_or(0.0);

        // Compute Lorentzian function for each x value
        let result = x
            .iter()
            .map(|&x_val| {
                let diff = x_val - center;
                let denominator = diff * diff + gamma * gamma;
                amplitude * gamma * gamma / denominator + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
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
        let gamma = self
            .params
            .get(&format!("{}gamma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}gamma", self.prefix)))?
            .value();

        // Count the actual number of varying parameters
        let has_baseline = self
            .params
            .get(&format!("{}baseline", self.prefix))
            .is_some();
        let n_params = if has_baseline { 4 } else { 3 };

        // Initialize Jacobian matrix
        let mut jac = Array2::zeros((x.len(), n_params));

        // Compute partial derivatives for each x value and parameter
        for (i, &x_val) in x.iter().enumerate() {
            let diff = x_val - center;
            let denominator = diff * diff + gamma * gamma;
            let gamma_squared = gamma * gamma;

            // d/d(amplitude)
            jac[[i, 0]] = gamma_squared / denominator;

            // d/d(center)
            jac[[i, 1]] = amplitude * 2.0 * gamma_squared * diff / (denominator * denominator);

            // d/d(gamma)
            jac[[i, 2]] = amplitude * 2.0 * gamma * (diff * diff - gamma_squared)
                / (denominator * denominator);

            // d/d(baseline) if applicable
            if has_baseline {
                jac[[i, 3]] = 1.0;
            }
        }

        Ok(jac)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if !self.with_init {
            return Ok(());
        }

        if x.len() != y.len() || x.is_empty() {
            return Err(LmOptError::DimensionMismatch(
                "x and y must have the same non-zero length".to_string(),
            ));
        }

        // Find the maximum value in y and its corresponding x
        let (max_idx, &max_y) = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| {
                LmOptError::ComputationError("Failed to find maximum y value".to_string())
            })?;

        let max_x = x[max_idx];

        // Find the minimum value for baseline estimation
        let min_y = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Estimate amplitude as the height above baseline
        let amplitude_est = max_y - min_y;

        // Estimate center as the x value at the maximum y
        let center_est = max_x;

        // Estimate gamma by checking width around center point
        // Look for points where y falls to half the peak height above baseline
        let half_height = min_y + amplitude_est / 2.0;

        // Find points to the left and right of center that cross half height
        let mut left_idx = max_idx;
        while left_idx > 0 && y[left_idx] > half_height {
            left_idx -= 1;
        }

        let mut right_idx = max_idx;
        while right_idx < y.len() - 1 && y[right_idx] > half_height {
            right_idx += 1;
        }

        // Estimate FWHM from these points
        let fwhm_est = if left_idx < right_idx {
            (x[right_idx] - x[left_idx]).abs()
        } else {
            // Default if we couldn't find crossings
            (x[x.len() - 1] - x[0]).abs() / 5.0
        };

        // Convert FWHM to gamma: gamma = FWHM / 2
        let gamma_est = fwhm_est / 2.0;

        // Update parameters with estimates
        let amp_param = self
            .params
            .get_mut(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?;
        amp_param.set_value(amplitude_est.max(0.0))?;

        let center_param = self
            .params
            .get_mut(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?;
        center_param.set_value(center_est)?;

        let gamma_param = self
            .params
            .get_mut(&format!("{}gamma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}gamma", self.prefix)))?;
        gamma_param.set_value(gamma_est.max(0.1))?;

        // Set baseline if applicable
        if let Some(baseline_param) = self.params.get_mut(&format!("{}baseline", self.prefix)) {
            baseline_param.set_value(min_y)?;
        }

        // The derived parameter fwhm will be updated automatically when the parameters
        // are used since it uses an expression

        Ok(())
    }
}

/// A Voigt peak model, which is a convolution of Gaussian and Lorentzian.
///
/// Since the actual Voigt function involves a complex error function that's
/// computationally expensive, this implements a pseudo-Voigt approximation
/// which is a weighted sum of Gaussian and Lorentzian components:
///
/// f(x) = amplitude * [fraction * lorentzian(x) + (1-fraction) * gaussian(x)] + baseline
///
/// Parameters:
/// - `amplitude`: Overall peak height
/// - `center`: Peak position
/// - `sigma`: Gaussian width parameter (standard deviation)
/// - `gamma`: Lorentzian width parameter (HWHM)
/// - `fraction`: Mixing fraction between Lorentzian and Gaussian (0 to 1)
/// - `baseline`: Baseline offset
#[derive(Debug, Clone)]
pub struct PseudoVoigtModel {
    params: Parameters,
    prefix: String,
    with_init: bool,
}

impl PseudoVoigtModel {
    /// Create a new PseudoVoigt model.
    ///
    /// # Arguments
    ///
    /// * `prefix` - Prefix for parameter names
    /// * `with_baseline` - Whether to include a baseline parameter
    ///
    /// # Returns
    ///
    /// * A new PseudoVoigtModel instance
    pub fn new(prefix: &str, with_baseline: bool) -> Self {
        let mut params = Parameters::new();

        // Add parameters with default values and bounds
        let amplitude_name = format!("{}amplitude", prefix);
        let center_name = format!("{}center", prefix);
        let sigma_name = format!("{}sigma", prefix);
        let gamma_name = format!("{}gamma", prefix);
        let fraction_name = format!("{}fraction", prefix);
        let baseline_name = format!("{}baseline", prefix);

        params
            .add_param_with_bounds(&amplitude_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();
        params.add_param(&center_name, 0.0).unwrap();
        params
            .add_param_with_bounds(&sigma_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();
        params
            .add_param_with_bounds(&gamma_name, 1.0, 0.0, f64::INFINITY)
            .unwrap();
        params
            .add_param_with_bounds(&fraction_name, 0.5, 0.0, 1.0)
            .unwrap();

        if with_baseline {
            params.add_param(&baseline_name, 0.0).unwrap();
        }

        // Add a derived parameter for Gaussian FWHM
        let g_fwhm_name = format!("{}g_fwhm", prefix);
        let sigma_expr = format!("{}sigma", prefix);
        params
            .add_param_with_expr(&g_fwhm_name, 2.3548, &format!("2.3548 * {}", sigma_expr))
            .unwrap();

        // Add a derived parameter for Lorentzian FWHM
        let l_fwhm_name = format!("{}l_fwhm", prefix);
        let gamma_expr = format!("{}gamma", prefix);
        params
            .add_param_with_expr(&l_fwhm_name, 2.0, &format!("2.0 * {}", gamma_expr))
            .unwrap();

        Self {
            params,
            prefix: prefix.to_string(),
            with_init: true, // Default to auto-initialization
        }
    }

    /// Set whether to auto-initialize parameters based on data.
    pub fn with_initialization(mut self, init: bool) -> Self {
        self.with_init = init;
        self
    }
}

impl Model for PseudoVoigtModel {
    fn parameters(&self) -> &Parameters {
        &self.params
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.params
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
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
        let gamma = self
            .params
            .get(&format!("{}gamma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}gamma", self.prefix)))?
            .value();
        let fraction = self
            .params
            .get(&format!("{}fraction", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}fraction", self.prefix)))?
            .value();

        // Get baseline if it exists
        let baseline = self
            .params
            .get(&format!("{}baseline", self.prefix))
            .map(|p| p.value())
            .unwrap_or(0.0);

        // Compute PseudoVoigt function for each x value
        let result = x
            .iter()
            .map(|&x_val| {
                // Gaussian component
                let g_arg = (x_val - center) / sigma;
                let gaussian = (-0.5 * g_arg * g_arg).exp();

                // Lorentzian component
                let diff = x_val - center;
                let denominator = diff * diff + gamma * gamma;
                let lorentzian = gamma * gamma / denominator;

                // Weighted sum and scaling
                amplitude * (fraction * lorentzian + (1.0 - fraction) * gaussian) + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    }

    fn jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>> {
        // For PseudoVoigt, analytical Jacobian is complex
        // Using numerical differentiation is more robust
        Err(LmOptError::NotImplemented(
            "Analytical Jacobian not implemented for PseudoVoigtModel".to_string(),
        ))
    }

    fn has_custom_jacobian(&self) -> bool {
        false
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        if !self.with_init {
            return Ok(());
        }

        if x.len() != y.len() || x.is_empty() {
            return Err(LmOptError::DimensionMismatch(
                "x and y must have the same non-zero length".to_string(),
            ));
        }

        // Find the maximum value in y and its corresponding x
        let (max_idx, &max_y) = y
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or_else(|| {
                LmOptError::ComputationError("Failed to find maximum y value".to_string())
            })?;

        let max_x = x[max_idx];

        // Find the minimum value for baseline estimation
        let min_y = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Estimate amplitude as the height above baseline
        let amplitude_est = max_y - min_y;

        // Estimate center as the x value at the maximum y
        let center_est = max_x;

        // Estimate width by checking width around center point
        // Look for points where y falls to half the peak height above baseline
        let half_height = min_y + amplitude_est / 2.0;

        // Find points to the left and right of center that cross half height
        let mut left_idx = max_idx;
        while left_idx > 0 && y[left_idx] > half_height {
            left_idx -= 1;
        }

        let mut right_idx = max_idx;
        while right_idx < y.len() - 1 && y[right_idx] > half_height {
            right_idx += 1;
        }

        // Estimate FWHM from these points
        let fwhm_est = if left_idx < right_idx {
            (x[right_idx] - x[left_idx]).abs()
        } else {
            // Default if we couldn't find crossings
            (x[x.len() - 1] - x[0]).abs() / 5.0
        };

        // Convert FWHM to sigma and gamma (roughly equal for a 50/50 mix)
        let sigma_est = fwhm_est / 2.3548;
        let gamma_est = fwhm_est / 2.0;

        // Update parameters with estimates
        let amp_param = self
            .params
            .get_mut(&format!("{}amplitude", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}amplitude", self.prefix)))?;
        amp_param.set_value(amplitude_est.max(0.0))?;

        let center_param = self
            .params
            .get_mut(&format!("{}center", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}center", self.prefix)))?;
        center_param.set_value(center_est)?;

        let sigma_param = self
            .params
            .get_mut(&format!("{}sigma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}sigma", self.prefix)))?;
        sigma_param.set_value(sigma_est.max(0.1))?;

        let gamma_param = self
            .params
            .get_mut(&format!("{}gamma", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}gamma", self.prefix)))?;
        gamma_param.set_value(gamma_est.max(0.1))?;

        // Start with equal mix of Gaussian and Lorentzian
        let fraction_param = self
            .params
            .get_mut(&format!("{}fraction", self.prefix))
            .ok_or_else(|| LmOptError::ParameterNotFound(format!("{}fraction", self.prefix)))?;
        fraction_param.set_value(0.5)?;

        // Set baseline if applicable
        if let Some(baseline_param) = self.params.get_mut(&format!("{}baseline", self.prefix)) {
            baseline_param.set_value(min_y)?;
        }

        // The derived parameters g_fwhm and l_fwhm will be updated automatically
        // when the parameters are used since they use expressions

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_gaussian_model() {
        // Create model
        let mut gaussian = GaussianModel::new("g_", true);

        // Set parameters
        gaussian
            .parameters_mut()
            .get_mut("g_amplitude")
            .unwrap()
            .set_value(2.0)
            .unwrap();
        gaussian
            .parameters_mut()
            .get_mut("g_center")
            .unwrap()
            .set_value(0.0)
            .unwrap();
        gaussian
            .parameters_mut()
            .get_mut("g_sigma")
            .unwrap()
            .set_value(1.0)
            .unwrap();
        gaussian
            .parameters_mut()
            .get_mut("g_baseline")
            .unwrap()
            .set_value(0.5)
            .unwrap();

        // Check FWHM derived parameter
        let fwhm = gaussian.parameters().get("g_fwhm").unwrap().value();
        assert_relative_eq!(fwhm, 2.3548, epsilon = 1e-4);

        // Test evaluation
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = gaussian.eval(&x).unwrap();

        // Expected values: 2.0 * exp(-x^2/2) + 0.5
        let expected = vec![
            2.0 * (-2.0f64.powi(2) / 2.0).exp() + 0.5,
            2.0 * (-1.0f64.powi(2) / 2.0).exp() + 0.5,
            2.0 * (0.0f64.powi(2) / 2.0).exp() + 0.5,
            2.0 * (-1.0f64.powi(2) / 2.0).exp() + 0.5,
            2.0 * (-2.0f64.powi(2) / 2.0).exp() + 0.5,
        ];

        assert_eq!(y.len(), 5);
        for i in 0..5 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_lorentzian_model() {
        // Create model
        let mut lorentzian = LorentzianModel::new("l_", true);

        // Set parameters
        lorentzian
            .parameters_mut()
            .get_mut("l_amplitude")
            .unwrap()
            .set_value(2.0)
            .unwrap();
        lorentzian
            .parameters_mut()
            .get_mut("l_center")
            .unwrap()
            .set_value(0.0)
            .unwrap();
        lorentzian
            .parameters_mut()
            .get_mut("l_gamma")
            .unwrap()
            .set_value(1.0)
            .unwrap();
        lorentzian
            .parameters_mut()
            .get_mut("l_baseline")
            .unwrap()
            .set_value(0.5)
            .unwrap();

        // Check FWHM derived parameter
        let fwhm = lorentzian.parameters().get("l_fwhm").unwrap().value();
        assert_relative_eq!(fwhm, 2.0, epsilon = 1e-10);

        // Test evaluation
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = lorentzian.eval(&x).unwrap();

        // Expected values: 2.0 * (1^2 / (x^2 + 1^2)) + 0.5
        let expected = vec![
            2.0 * (1.0 / (4.0 + 1.0)) + 0.5,
            2.0 * (1.0 / (1.0 + 1.0)) + 0.5,
            2.0 * (1.0 / (0.0 + 1.0)) + 0.5,
            2.0 * (1.0 / (1.0 + 1.0)) + 0.5,
            2.0 * (1.0 / (4.0 + 1.0)) + 0.5,
        ];

        assert_eq!(y.len(), 5);
        for i in 0..5 {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parameter_initialization() {
        // Create synthetic data with a Gaussian peak
        let x = Array1::linspace(-10.0, 10.0, 100);
        let y: Array1<f64> = x
            .iter()
            .map(|&x_val| {
                let arg = -((x_val - 2.0) * (x_val - 2.0)) / (2.0 * 1.5 * 1.5);
                3.0 * f64::exp(arg) + 0.5
            })
            .collect();

        // Create model and initialize parameters
        let mut gaussian = GaussianModel::new("g_", true);
        gaussian.guess_parameters(&x, &y).unwrap();

        // Check that parameters are close to the true values
        let amplitude = gaussian.parameters().get("g_amplitude").unwrap().value();
        let center = gaussian.parameters().get("g_center").unwrap().value();
        let sigma = gaussian.parameters().get("g_sigma").unwrap().value();
        let baseline = gaussian.parameters().get("g_baseline").unwrap().value();

        println!(
            "Estimated parameters: amplitude={}, center={}, sigma={}, baseline={}",
            amplitude, center, sigma, baseline
        );

        assert_relative_eq!(amplitude, 3.0, epsilon = 0.5);
        assert_relative_eq!(center, 2.0, epsilon = 0.5);
        assert_relative_eq!(sigma, 1.5, epsilon = 0.5);
        assert_relative_eq!(baseline, 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_pseudovoigt_model() {
        // Create model
        let mut voigt = PseudoVoigtModel::new("v_", true);

        // Set parameters
        voigt
            .parameters_mut()
            .get_mut("v_amplitude")
            .unwrap()
            .set_value(2.0)
            .unwrap();
        voigt
            .parameters_mut()
            .get_mut("v_center")
            .unwrap()
            .set_value(0.0)
            .unwrap();
        voigt
            .parameters_mut()
            .get_mut("v_sigma")
            .unwrap()
            .set_value(1.0)
            .unwrap();
        voigt
            .parameters_mut()
            .get_mut("v_gamma")
            .unwrap()
            .set_value(1.0)
            .unwrap();
        voigt
            .parameters_mut()
            .get_mut("v_fraction")
            .unwrap()
            .set_value(0.5)
            .unwrap();
        voigt
            .parameters_mut()
            .get_mut("v_baseline")
            .unwrap()
            .set_value(0.5)
            .unwrap();

        // Test evaluation
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = voigt.eval(&x).unwrap();

        // Expected values are a mix of Gaussian and Lorentzian
        // Here we just check that the function is symmetric around 0
        assert_eq!(y.len(), 5);
        assert_relative_eq!(y[0], y[4], epsilon = 1e-10); // x = -2 and x = 2
        assert_relative_eq!(y[1], y[3], epsilon = 1e-10); // x = -1 and x = 1
        assert!(y[2] > y[1]); // Peak at center
    }
}
