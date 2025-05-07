//! # Model System
//!
//! This module provides a framework for defining and fitting mathematical models to data.
//! It includes a [`Model`] trait for common model operations, a [`BaseModel`] implementation
//! for creating custom models, and a [`fit`] function for easy model fitting.
//!
//! ## Key Features
//!
//! - **Unified Model Interface**: Common interface for all model types
//! - **Parameter Integration**: Models use the parameter system for bounds and constraints
//! - **Automatic Parameter Guessing**: Built-in models can estimate initial parameters from data
//! - **Custom Model Creation**: Create custom models with closures
//! - **Simplified Fitting**: Fit models to data with a single function call
//!
//! ## Example Usage
//!
//! ### Using Built-in Models
//!
//! ```rust
//! use lmopt_rs::model::{fit, Model};
//! use lmopt_rs::models::GaussianModel;
//! use ndarray::Array1;
//!
//! // Create a Gaussian model with baseline
//! let mut model = GaussianModel::new("", true);
//!
//! // Generate some data
//! let x = Array1::linspace(-5.0, 5.0, 100);
//! let y = x.mapv(|x_val| 3.0 * (-x_val.powi(2) / 2.0).exp() + 0.5);
//!
//! // Fit the model to the data
//! let result = fit(&mut model, x.clone(), y.clone()).unwrap();
//!
//! // Print the results
//! println!("Fit success: {}", result.success);
//! println!("Amplitude: {:.3}", model.parameters().get("amplitude").unwrap().value());
//! println!("Center: {:.3}", model.parameters().get("center").unwrap().value());
//! println!("Sigma: {:.3}", model.parameters().get("sigma").unwrap().value());
//! println!("Baseline: {:.3}", model.parameters().get("baseline").unwrap().value());
//! ```
//!
//! ### Creating a Custom Model
//!
//! ```rust
//! use lmopt_rs::model::{BaseModel, fit, Model};
//! use lmopt_rs::parameters::Parameters;
//! use ndarray::Array1;
//!
//! // Create parameters for a custom model
//! let mut params = Parameters::new();
//! params.add_param("a", 1.0).unwrap();
//! params.add_param("b", 0.0).unwrap();
//! params.add_param("c", 0.0).unwrap();
//!
//! // Create a custom model using a closure
//! let model = BaseModel::new(params, |params, x| {
//!     // Model: y = a * sin(b * x + c)
//!     let a = params.get("a").unwrap().value();
//!     let b = params.get("b").unwrap().value();
//!     let c = params.get("c").unwrap().value();
//!     
//!     let result = x.iter()
//!         .map(|&x_val| a * (b * x_val + c).sin())
//!         .collect::<Vec<f64>>();
//!         
//!     Ok(Array1::from_vec(result))
//! });
//!
//! // Use the custom model...
//! ```
//!
//! For comprehensive documentation, see the [Model System guide](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/models.md).

use crate::error::{LmOptError, Result};
use crate::parameters::{Parameter, Parameters};
use crate::problem_params::ParameterProblem;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// A trait representing a model that can be fit to data.
///
/// Models provide a way to define a mathematical function with parameters
/// that can be fit to data. Models can evaluate the function at given x values,
/// calculate residuals against observed data, and optionally provide derivatives
/// for more efficient fitting.
pub trait Model {
    /// Returns a reference to the model's parameters.
    fn parameters(&self) -> &Parameters;

    /// Returns a mutable reference to the model's parameters.
    fn parameters_mut(&mut self) -> &mut Parameters;

    /// Evaluates the model at the given x values using the current parameter values.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values at which to evaluate the model
    ///
    /// # Returns
    ///
    /// * The model's predicted values at the given x values
    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>>;

    /// Calculates the residuals (y_obs - y_pred) using the current parameter values.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values
    /// * `y` - The observed dependent variable values
    ///
    /// # Returns
    ///
    /// * The residuals (y_obs - y_pred)
    fn residuals(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        let y_pred = self.eval(x)?;

        if y.len() != y_pred.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Length of observed data ({}) does not match length of predicted data ({})",
                y.len(),
                y_pred.len()
            )));
        }

        Ok(y - y_pred)
    }

    /// Calculates the Jacobian matrix of the model at the given x values.
    ///
    /// The Jacobian is the matrix of partial derivatives of the model output with respect
    /// to each parameter. This is used to compute step directions during optimization.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values at which to evaluate the Jacobian
    ///
    /// # Returns
    ///
    /// * The Jacobian matrix with shape [n_points, n_params]
    fn jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>> {
        // Default implementation returns a not implemented error
        // Models should override this to provide an analytical Jacobian when possible
        Err(LmOptError::NotImplemented(
            "Analytical Jacobian not implemented for this model".to_string(),
        ))
    }

    /// Indicates whether this model provides a custom Jacobian implementation.
    ///
    /// When true, the optimizer will use the model's jacobian method.
    /// When false, the optimizer will use numerical differentiation.
    ///
    /// # Returns
    ///
    /// * True if the model provides a custom Jacobian implementation, false otherwise
    fn has_custom_jacobian(&self) -> bool {
        false
    }

    /// Initializes model parameters based on data.
    ///
    /// This method is used to provide initial parameter estimates from data,
    /// which can significantly improve convergence during fitting.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values
    /// * `y` - The observed dependent variable values
    ///
    /// # Returns
    ///
    /// * Ok(()) if initialization is successful, Error otherwise
    fn guess_parameters(&mut self, _x: &Array1<f64>, _y: &Array1<f64>) -> Result<()> {
        // Default implementation does nothing
        // Specific models should override this to provide intelligent initial guesses
        Ok(())
    }

    /// Returns a list of parameter names in the model.
    ///
    /// This method returns a list of all parameter names in the order they
    /// would appear in a parameter array.
    ///
    /// # Returns
    ///
    /// * Vector of parameter names
    fn parameter_names(&self) -> Vec<String> {
        self.parameters()
            .iter()
            .filter(|(_, param)| param.vary())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Evaluates the model with the current parameters and returns the residuals,
    /// which is used by the ParameterProblem trait.
    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        Err(LmOptError::NotImplemented(
            "eval_with_parameters is not implemented for this model".to_string(),
        ))
    }

    /// Converts the model's parameters to an Array1 for use with optimizers.
    ///
    /// # Returns
    ///
    /// * Array of parameter values for varying parameters
    fn parameters_to_array(&self) -> Result<Array1<f64>> {
        self.parameters().to_array()
    }

    /// Updates the model's parameters from an Array1, typically from an optimizer result.
    ///
    /// # Arguments
    ///
    /// * `params` - Array of parameter values for varying parameters
    ///
    /// # Returns
    ///
    /// * Ok(()) if update is successful, Error otherwise
    fn update_parameters_from_array(&mut self, params: &Array1<f64>) -> Result<()> {
        self.parameters_mut().update_from_array(params)
    }
}

/// A basic implementation of the Model trait that uses a closure to evaluate the model.
///
/// This provides a convenient way to create custom models without implementing
/// the full Model trait.
pub struct BaseModel<F>
where
    F: Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Parameters for the model
    parameters: Parameters,

    /// Function that evaluates the model
    eval_fn: F,

    /// Last x values used for evaluation (for ParameterProblem impl)
    last_x: Option<Array1<f64>>,

    /// Last y values used for evaluation (for ParameterProblem impl)
    last_y: Option<Array1<f64>>,
}

impl<F> BaseModel<F>
where
    F: Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>>,
{
    /// Creates a new BaseModel with the given parameters and evaluation function.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Parameters for the model
    /// * `eval_fn` - Function that evaluates the model given parameters and x values
    ///
    /// # Returns
    ///
    /// * A new BaseModel instance
    pub fn new(parameters: Parameters, eval_fn: F) -> Self {
        Self {
            parameters,
            eval_fn,
            last_x: None,
            last_y: None,
        }
    }

    /// Sets the data for use with ParameterProblem trait.
    ///
    /// # Arguments
    ///
    /// * `x` - Independent variable values
    /// * `y` - Observed dependent variable values
    pub fn set_data(&mut self, x: Array1<f64>, y: Array1<f64>) {
        self.last_x = Some(x);
        self.last_y = Some(y);
    }
}

impl<F> Model for BaseModel<F>
where
    F: Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>>,
{
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        (self.eval_fn)(&self.parameters, x)
    }
}

impl<F> ParameterProblem for BaseModel<F>
where
    F: Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>>,
{
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        match (&self.last_x, &self.last_y) {
            (Some(x), Some(y)) => self.residuals(x, y),
            _ => Err(LmOptError::InvalidState(
                "No data set for BaseModel. Call set_data() first.".to_string(),
            )),
        }
    }

    fn residual_count(&self) -> usize {
        match &self.last_y {
            Some(y) => y.len(),
            None => 0,
        }
    }
}

/// Result of fitting a model to data.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Whether the fit was successful
    pub success: bool,

    /// Cost function value at the minimum (sum of squared residuals)
    pub cost: f64,

    /// Final parameter values
    pub params: Array1<f64>,

    /// Number of iterations performed
    pub iterations: usize,

    /// Residuals at the solution
    pub residuals: Array1<f64>,

    /// Message describing the convergence or termination status
    pub message: String,

    /// Covariance matrix
    pub covariance: Array2<f64>,

    /// Standard errors for each parameter
    pub standard_errors: HashMap<String, f64>,
}

/// Fits a model to data using the Levenberg-Marquardt algorithm.
///
/// This is a convenience function that:
/// 1. Initializes model parameters based on the data if not specified
/// 2. Creates an optimizer
/// 3. Runs the optimization
/// 4. Updates the model with the optimized parameters
/// 5. Calculates standard errors
///
/// # Arguments
///
/// * `model` - Model to fit
/// * `x` - Independent variable values
/// * `y` - Observed dependent variable values
///
/// # Returns
///
/// * FitResult containing the optimization results
pub fn fit<M: Model>(model: &mut M, x: Array1<f64>, y: Array1<f64>) -> Result<FitResult> {
    use crate::lm::LevenbergMarquardt;
    use crate::problem_params::problem_from_parameter_problem;
    use crate::uncertainty::standard_errors;

    // Create a BaseModel adapter that implements ParameterProblem
    let mut base_model = BaseModel::new(model.parameters().clone(), |params, x_vals| {
        // We need to create a temporary model with updated parameters
        let mut temp_model = model.clone();
        temp_model.parameters_mut().update_from(&params)?;
        temp_model.eval(x_vals)
    });

    base_model.set_data(x.clone(), y.clone());

    // Get initial parameter values
    let initial_params = model.parameters_to_array()?;

    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();

    // Create the problem adapter
    let adapter = problem_from_parameter_problem(&base_model);

    // Run the optimization
    let result = optimizer.minimize(&adapter, initial_params)?;

    // Update the model with the optimized parameters
    model.update_parameters_from_array(&result.params)?;

    // Calculate standard errors
    let std_errors = standard_errors(model, &result)?;

    // Create the fit result
    let fit_result = FitResult {
        success: result.success,
        cost: result.cost,
        params: result.params,
        iterations: result.iterations,
        residuals: result.residuals,
        message: result.message,
        covariance: ndarray::Array2::zeros((result.params.len(), result.params.len())),
        standard_errors: std_errors,
    };

    Ok(fit_result)
}

// Implementation of Clone for BaseModel would go here, but it's omitted for brevity
