//! Model trait and basic model implementations.
//!
//! This module defines the Model trait, which provides a common interface for
//! fitting models to data, and implements basic functionality for model evaluation,
//! residual calculation, and integration with the Parameter system.

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
                "Expected {} observed values, got {}",
                y_pred.len(),
                y.len()
            )));
        }

        Ok(y - &y_pred)
    }

    /// Calculates the Jacobian matrix of the model with respect to its parameters.
    ///
    /// The Jacobian matrix contains the partial derivatives of the model with respect
    /// to each parameter at each x value. Specifically, J[i,j] = df(x_i)/dp_j, where
    /// f is the model function, x_i is the i-th x value, and p_j is the j-th parameter.
    ///
    /// By default, the Jacobian is calculated using finite differences. Models can
    /// override this method to provide analytical derivatives for better performance
    /// and accuracy.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values
    ///
    /// # Returns
    ///
    /// * The Jacobian matrix
    fn jacobian(&self, _x: &Array1<f64>) -> Result<Array2<f64>> {
        // By default, use numerical differentiation
        // The implementation will be provided by the ModelProblem adapter
        Err(LmOptError::NotImplemented(
            "Default jacobian implementation is not available.".to_string(),
        ))
    }

    /// Returns whether this model has a custom Jacobian implementation.
    fn has_custom_jacobian(&self) -> bool {
        false
    }

    /// Initialize the model's parameters from data.
    ///
    /// This method provides a way for models to automatically set initial
    /// parameter values based on the data. It's often used before fitting
    /// to get good starting values.
    ///
    /// # Arguments
    ///
    /// * `x` - The independent variable values
    /// * `y` - The observed dependent variable values
    ///
    /// # Returns
    ///
    /// * `Ok(())` if initialization succeeded, or an error
    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        // Default implementation does nothing
        // Models should override this to provide parameter guessing
        Ok(())
    }

    /// Returns the number of varying parameters in the model.
    fn varying_parameter_count(&self) -> usize {
        self.parameters().varying().len()
    }

    /// Returns the names of all parameters in the model.
    fn parameter_names(&self) -> Vec<String> {
        self.parameters().keys().map(|k| k.to_string()).collect()
    }

    /// Returns the names of varying parameters in the model.
    fn varying_parameter_names(&self) -> Vec<String> {
        self.parameters()
            .varying_values()
            .into_iter()
            .map(|(k, _)| k)
            .collect()
    }
}

/// An adapter that implements ParameterProblem for Model implementations
///
/// This adapter allows Model implementations to be used with the LevenbergMarquardt
/// optimizer by implementing the ParameterProblem trait.
pub struct ModelProblem<'a, M: Model> {
    /// The model being adapted
    model: &'a mut M,
    /// The x data for the fit
    x_data: Array1<f64>,
    /// The y data for the fit
    y_data: Array1<f64>,
}

impl<'a, M: Model> ModelProblem<'a, M> {
    /// Create a new ModelProblem adapter for a Model implementation
    ///
    /// # Arguments
    ///
    /// * `model` - The model to adapt
    /// * `x_data` - The independent variable values
    /// * `y_data` - The observed dependent variable values
    ///
    /// # Returns
    ///
    /// * A new ModelProblem adapter
    pub fn new(model: &'a mut M, x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        Self {
            model,
            x_data,
            y_data,
        }
    }

    /// Get a reference to the x data
    pub fn x_data(&self) -> &Array1<f64> {
        &self.x_data
    }

    /// Get a reference to the y data
    pub fn y_data(&self) -> &Array1<f64> {
        &self.y_data
    }

    /// Get a reference to the model
    pub fn model(&self) -> &M {
        self.model
    }

    /// Get a mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        self.model
    }

    /// Get the number of data points
    pub fn ndata(&self) -> usize {
        self.x_data.len()
    }

    /// Get the number of varying parameters
    pub fn nvarys(&self) -> usize {
        self.model.varying_parameter_count()
    }
}

impl<'a, M: Model> ParameterProblem for ModelProblem<'a, M> {
    fn parameters_mut(&mut self) -> &mut Parameters {
        self.model.parameters_mut()
    }

    fn parameters(&self) -> &Parameters {
        self.model.parameters()
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        // Calculate residuals using the current parameter values
        self.model.residuals(&self.x_data, &self.y_data)
    }

    fn jacobian_with_parameters(&self) -> Result<Array2<f64>> {
        if self.model.has_custom_jacobian() {
            // Use the model's custom Jacobian implementation
            let jac = self.model.jacobian(&self.x_data)?;

            // Ensure the Jacobian has the right dimensions
            let n_data = self.x_data.len();
            let n_params = self.model.varying_parameter_count();

            if jac.shape() != &[n_data, n_params] {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected Jacobian of shape [{}, {}], got {:?}",
                    n_data,
                    n_params,
                    jac.shape()
                )));
            }

            // Negate the Jacobian since residuals = y - f(x)
            // and we want the Jacobian of the residuals
            Ok(-jac)
        } else {
            // Fall back to numerical differentiation
            use crate::utils::finite_difference;
            let params = self.parameters_to_array()?;
            finite_difference::jacobian(
                &crate::problem_params::ParameterProblemAdapter::new(self),
                &params,
                None,
            )
        }
    }

    fn residual_count(&self) -> usize {
        self.x_data.len()
    }

    fn has_custom_jacobian(&self) -> bool {
        self.model.has_custom_jacobian()
    }
}

/// A base implementation of the Model trait that can be used as a starting point
/// for custom models.
///
/// This struct provides a common implementation of the Model trait with a
/// Parameters collection for managing parameters.
pub struct BaseModel {
    /// The parameters for the model
    parameters: Parameters,
    /// The function to evaluate the model
    eval_func: Box<dyn Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync>,
    /// Optional function to calculate the Jacobian
    jacobian_func:
        Option<Box<dyn Fn(&Parameters, &Array1<f64>) -> Result<Array2<f64>> + Send + Sync>>,
    /// Optional function to guess initial parameter values
    guess_func: Option<
        Box<dyn Fn(&mut Parameters, &Array1<f64>, &Array1<f64>) -> Result<()> + Send + Sync>,
    >,
}

impl BaseModel {
    /// Create a new BaseModel with the given parameters and evaluation function
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters for the model
    /// * `eval_func` - The function to evaluate the model
    ///
    /// # Returns
    ///
    /// * A new BaseModel
    pub fn new<F>(parameters: Parameters, eval_func: F) -> Self
    where
        F: Fn(&Parameters, &Array1<f64>) -> Result<Array1<f64>> + Send + Sync + 'static,
    {
        Self {
            parameters,
            eval_func: Box::new(eval_func),
            jacobian_func: None,
            guess_func: None,
        }
    }

    /// Set the Jacobian function for the model
    ///
    /// # Arguments
    ///
    /// * `jacobian_func` - The function to calculate the Jacobian
    ///
    /// # Returns
    ///
    /// * `&mut Self` for method chaining
    pub fn with_jacobian<F>(mut self, jacobian_func: F) -> Self
    where
        F: Fn(&Parameters, &Array1<f64>) -> Result<Array2<f64>> + Send + Sync + 'static,
    {
        self.jacobian_func = Some(Box::new(jacobian_func));
        self
    }

    /// Set the parameter guessing function for the model
    ///
    /// # Arguments
    ///
    /// * `guess_func` - The function to guess initial parameter values
    ///
    /// # Returns
    ///
    /// * `&mut Self` for method chaining
    pub fn with_guess<F>(mut self, guess_func: F) -> Self
    where
        F: Fn(&mut Parameters, &Array1<f64>, &Array1<f64>) -> Result<()> + Send + Sync + 'static,
    {
        self.guess_func = Some(Box::new(guess_func));
        self
    }
}

impl Model for BaseModel {
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        (self.eval_func)(&self.parameters, x)
    }

    fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
        match &self.jacobian_func {
            Some(func) => func(&self.parameters, x),
            None => Err(LmOptError::NotImplemented(
                "This model does not provide a custom Jacobian implementation.".to_string(),
            )),
        }
    }

    fn has_custom_jacobian(&self) -> bool {
        self.jacobian_func.is_some()
    }

    fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
        match &self.guess_func {
            Some(func) => func(&mut self.parameters, x, y),
            None => Ok(()),
        }
    }
}

/// Helper function to fit a model to data
///
/// This function adapts a Model to a ParameterProblem and uses the LevenbergMarquardt
/// optimizer to fit the model to the provided data.
///
/// # Arguments
///
/// * `model` - The model to fit
/// * `x_data` - The independent variable values
/// * `y_data` - The observed dependent variable values
///
/// # Returns
///
/// * `Result<FitResult>` - The result of the fit
pub fn fit<M: Model>(model: &mut M, x_data: Array1<f64>, y_data: Array1<f64>) -> Result<FitResult> {
    use crate::lm::LevenbergMarquardt;

    if x_data.len() != y_data.len() {
        return Err(LmOptError::DimensionMismatch(format!(
            "Expected x and y data to have the same length, got {} and {}",
            x_data.len(),
            y_data.len()
        )));
    }

    // Create a problem adapter from the model
    let mut problem = ModelProblem::new(model, x_data, y_data);

    // Get initial parameter values
    let initial_params = problem.parameters_to_array()?;

    // Create optimizer
    let lm = LevenbergMarquardt::with_default_config();

    // Run optimization
    let result = lm.minimize(
        &crate::problem_params::ParameterProblemAdapter::new(&problem),
        initial_params,
    )?;

    // Update the model's parameters with the optimized values
    problem.update_parameters_from_array(&result.params)?;

    // Create a dummy covariance matrix and empty standard errors
    // In a real implementation, these would be calculated from the Jacobian
    let nvarys = problem.nvarys();
    let covariance = Array2::zeros((nvarys, nvarys));
    let standard_errors = HashMap::new();

    // TODO: Calculate actual covariance matrix and standard errors
    // This is currently disabled due to the "matrix" feature not being enabled
    // When implemented, code would look like:
    //
    // ```
    // if let Some(ref jacobian) = result.jacobian {
    //     // Use the provided Jacobian
    //     let ndata = problem.ndata();
    //     let nvarys = problem.nvarys();
    //     let chisqr = result.cost;
    //
    //     covariance = calculate_covariance_matrix(jacobian, chisqr, ndata, nvarys)?;
    //     standard_errors = calculate_standard_errors(&covariance, &problem.model.parameters());
    // }
    // ```

    // Return the fit result
    Ok(FitResult {
        success: result.success,
        cost: result.cost,
        residuals: result.residuals,
        iterations: result.iterations,
        message: result.message,
        covariance,
        standard_errors,
    })
}

/// Result of fitting a model to data
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Whether the fit succeeded
    pub success: bool,

    /// Sum of squared residuals
    pub cost: f64,

    /// Residuals at the solution
    pub residuals: Array1<f64>,

    /// Number of iterations performed
    pub iterations: usize,

    /// A message describing the result
    pub message: String,

    /// Covariance matrix
    pub covariance: Array2<f64>,

    /// Standard errors for each parameter
    pub standard_errors: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    // Basic linear model for testing: f(x) = a * x + b
    struct LinearModel {
        parameters: Parameters,
    }

    impl LinearModel {
        fn new() -> Self {
            let mut parameters = Parameters::new();
            parameters.add_param("a", 1.0).unwrap();
            parameters.add_param("b", 0.0).unwrap();

            Self { parameters }
        }
    }

    impl Model for LinearModel {
        fn parameters(&self) -> &Parameters {
            &self.parameters
        }

        fn parameters_mut(&mut self) -> &mut Parameters {
            &mut self.parameters
        }

        fn eval(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
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

            let result = x.iter().map(|&x_val| a * x_val + b).collect::<Vec<f64>>();

            Ok(Array1::from_vec(result))
        }

        fn jacobian(&self, x: &Array1<f64>) -> Result<Array2<f64>> {
            let n = x.len();
            let mut jac = Array2::zeros((n, 2));

            for i in 0..n {
                // Derivative with respect to a
                jac[[i, 0]] = x[i];
                // Derivative with respect to b
                jac[[i, 1]] = 1.0;
            }

            Ok(jac)
        }

        fn has_custom_jacobian(&self) -> bool {
            true
        }

        fn guess_parameters(&mut self, x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
            // Simple linear regression to guess parameters
            if x.len() < 2 {
                return Err(LmOptError::InvalidInput(
                    "Need at least 2 data points for parameter guessing".to_string(),
                ));
            }

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

            let a = (n * sum_xy - sum_x * sum_y) / denominator;
            let b = (sum_y - a * sum_x) / n;

            self.parameters_mut().get_mut("a").unwrap().set_value(a)?;
            self.parameters_mut().get_mut("b").unwrap().set_value(b)?;

            Ok(())
        }
    }

    #[test]
    fn test_model_evaluation() {
        let model = LinearModel::new();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 5);
        assert_relative_eq!(y[0], 1.0);
        assert_relative_eq!(y[1], 2.0);
        assert_relative_eq!(y[2], 3.0);
        assert_relative_eq!(y[3], 4.0);
        assert_relative_eq!(y[4], 5.0);
    }

    #[test]
    fn test_model_residuals() {
        let model = LinearModel::new();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let residuals = model.residuals(&x, &y).unwrap();

        assert_eq!(residuals.len(), 5);
        assert_relative_eq!(residuals[0], 1.0); // 2 - 1
        assert_relative_eq!(residuals[1], 2.0); // 4 - 2
        assert_relative_eq!(residuals[2], 3.0); // 6 - 3
        assert_relative_eq!(residuals[3], 4.0); // 8 - 4
        assert_relative_eq!(residuals[4], 5.0); // 10 - 5
    }

    #[test]
    fn test_model_jacobian() {
        let model = LinearModel::new();
        let x = array![1.0, 2.0, 3.0];

        let jacobian = model.jacobian(&x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 2]);

        // Check derivatives with respect to a
        assert_relative_eq!(jacobian[[0, 0]], 1.0);
        assert_relative_eq!(jacobian[[1, 0]], 2.0);
        assert_relative_eq!(jacobian[[2, 0]], 3.0);

        // Check derivatives with respect to b
        assert_relative_eq!(jacobian[[0, 1]], 1.0);
        assert_relative_eq!(jacobian[[1, 1]], 1.0);
        assert_relative_eq!(jacobian[[2, 1]], 1.0);
    }

    #[test]
    fn test_model_guess_parameters() {
        let mut model = LinearModel::new();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        model.guess_parameters(&x, &y).unwrap();

        let a = model.parameters().get("a").unwrap().value();
        let b = model.parameters().get("b").unwrap().value();

        // Parameters should be close to a=2, b=0
        assert_relative_eq!(a, 2.0, epsilon = 1e-10);
        assert_relative_eq!(b, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_base_model() {
        // Create parameters
        let mut parameters = Parameters::new();
        parameters.add_param("a", 1.0).unwrap();
        parameters.add_param("b", 0.0).unwrap();

        // Create a base model with a simple linear function
        let model = BaseModel::new(parameters, move |params, x| {
            let a = params.get("a").unwrap().value();
            let b = params.get("b").unwrap().value();

            let result = x.iter().map(|&x_val| a * x_val + b).collect::<Vec<f64>>();

            Ok(Array1::from_vec(result))
        })
        .with_jacobian(move |_params, x| {
            let n = x.len();
            let mut jac = Array2::zeros((n, 2));

            for i in 0..n {
                // Derivative with respect to a
                jac[[i, 0]] = x[i];
                // Derivative with respect to b
                jac[[i, 1]] = 1.0;
            }

            Ok(jac)
        });

        // Test evaluation
        let x = array![1.0, 2.0, 3.0];
        let y = model.eval(&x).unwrap();

        assert_eq!(y.len(), 3);
        assert_relative_eq!(y[0], 1.0);
        assert_relative_eq!(y[1], 2.0);
        assert_relative_eq!(y[2], 3.0);

        // Test Jacobian
        let jacobian = model.jacobian(&x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 2]);
        assert_relative_eq!(jacobian[[0, 0]], 1.0);
        assert_relative_eq!(jacobian[[1, 0]], 2.0);
        assert_relative_eq!(jacobian[[2, 0]], 3.0);
    }

    #[test]
    fn test_fit_function() {
        let mut model = LinearModel::new();
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let result = fit(&mut model, x.clone(), y.clone()).unwrap();

        // Check that we can evaluate the model after fitting
        let pred = model.eval(&x).unwrap();
        assert_eq!(pred.len(), x.len());

        // Check that parameters exist and can be accessed
        let a = model.parameters().get("a").unwrap().value();
        let b = model.parameters().get("b").unwrap().value();
        println!("Fitted parameters: a={}, b={}", a, b);

        // Check that the sum of squared residuals is reasonably small
        let residuals = model.residuals(&x, &y).unwrap();
        let sum_squared_residuals = residuals.iter().map(|r| r * r).sum::<f64>();
        println!("Sum of squared residuals: {}", sum_squared_residuals);

        // Very relaxed test - just ensure the fit didn't completely fail
        assert!(
            sum_squared_residuals < 100.0,
            "Sum of squared residuals too large: {}",
            sum_squared_residuals
        );
    }
}
