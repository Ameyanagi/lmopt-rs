//! Problem definition trait and implementations.
//!
//! This module defines the `Problem` trait, which represents a nonlinear
//! least squares problem to be solved with the Levenberg-Marquardt algorithm.
//! It also provides an adapter for compatibility with the levenberg-marquardt crate.

use ndarray::{Array1, Array2};
use crate::error::{LmOptError, Result};

/// A trait representing a nonlinear least squares problem.
///
/// This trait defines the interface for problems that can be solved using
/// the Levenberg-Marquardt algorithm.
pub trait Problem {
    /// Evaluate the residuals at the given parameters.
    ///
    /// This function calculates the vector of residuals (differences between the model
    /// and the data) at the given parameter values.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameter values at which to evaluate the residuals
    ///
    /// # Returns
    ///
    /// * A vector of residuals, or an error if the evaluation fails
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>>;
    
    /// Get the number of parameters in the problem.
    fn parameter_count(&self) -> usize;
    
    /// Get the number of residuals in the problem.
    fn residual_count(&self) -> usize;
    
    /// Evaluate the Jacobian matrix at the given parameters.
    ///
    /// The Jacobian is the matrix of partial derivatives of the residuals with respect
    /// to the parameters. The Jacobian is used to calculate the step direction.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameter values at which to evaluate the Jacobian
    ///
    /// # Returns
    ///
    /// * The Jacobian matrix, or an error if the evaluation fails
    ///
    /// # Default Implementation
    ///
    /// The default implementation uses automatic differentiation when possible,
    /// falling back to numerical differentiation otherwise.
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> where Self: Sized {
        // TODO: Implement using autodiff when possible, falling back to numerical differentiation
        crate::utils::finite_difference::jacobian(self, params, None)
    }
    
    /// Check if this problem provides a custom Jacobian implementation.
    ///
    /// If this returns true, the optimizer will use the `jacobian` method
    /// provided by the problem. If false, the optimizer may use an internal
    /// optimization to calculate the Jacobian more efficiently.
    fn has_custom_jacobian(&self) -> bool {
        false
    }
    
    /// Evaluate the sum of squared residuals at the given parameters.
    ///
    /// This function calculates the scalar cost function (sum of squared residuals)
    /// at the given parameter values. This is often used as a measure of goodness-of-fit.
    ///
    /// # Arguments
    ///
    /// * `params` - The parameter values at which to evaluate the cost
    ///
    /// # Returns
    ///
    /// * The cost value (sum of squared residuals), or an error if the evaluation fails
    fn eval_cost(&self, params: &Array1<f64>) -> Result<f64> {
        let residuals = self.eval(params)?;
        Ok(residuals.iter().map(|r| r.powi(2)).sum())
    }
}

/// Adapter for using the `levenberg-marquardt` crate's problem interface.
///
/// This adapter allows problems defined using the `levenberg-marquardt` crate's
/// `LeastSquaresProblem` trait to be used with our optimizer.
#[cfg(feature = "lm-compat")]
pub mod lm_compat {
    use super::*;
    use levenberg_marquardt::{LeastSquaresProblem, LeastSquaresFunc};
    use nalgebra::{DMatrix, DVector};
    use crate::utils::matrix_convert::{nalgebra_to_ndarray, ndarray_to_nalgebra};
    
    /// Adapter to use `levenberg-marquardt` crate's problems with our optimizer.
    pub struct LmProblemAdapter<P>
    where
        P: LeastSquaresProblem<f64, Params = DVector<f64>, Residuals = DVector<f64>, Jacobian = DMatrix<f64>>,
    {
        problem: P,
    }
    
    impl<P> LmProblemAdapter<P>
    where
        P: LeastSquaresProblem<f64, Params = DVector<f64>, Residuals = DVector<f64>, Jacobian = DMatrix<f64>>,
    {
        /// Create a new adapter for a `levenberg-marquardt` crate problem.
        pub fn new(problem: P) -> Self {
            Self { problem }
        }
    }
    
    impl<P> Problem for LmProblemAdapter<P>
    where
        P: LeastSquaresProblem<f64, Params = DVector<f64>, Residuals = DVector<f64>, Jacobian = DMatrix<f64>>,
    {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            // Convert ndarray params to nalgebra
            let nalgebra_params = ndarray_to_nalgebra(params)?;
            
            // Call the underlying problem's residuals method
            let nalgebra_residuals = self.problem.residuals(&nalgebra_params)
                .map_err(|e| LmOptError::FunctionEvaluation(format!("Error in LM problem residuals: {:?}", e)))?;
            
            // Convert nalgebra residuals to ndarray
            nalgebra_to_ndarray(&nalgebra_residuals)
        }
        
        fn parameter_count(&self) -> usize {
            self.problem.params_len()
        }
        
        fn residual_count(&self) -> usize {
            self.problem.residuals_len()
        }
        
        fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
            // Convert ndarray params to nalgebra
            let nalgebra_params = ndarray_to_nalgebra(params)?;
            
            // Call the underlying problem's jacobian method
            let nalgebra_jacobian = self.problem.jacobian(&nalgebra_params)
                .map_err(|e| LmOptError::FunctionEvaluation(format!("Error in LM problem jacobian: {:?}", e)))?;
            
            // Convert nalgebra jacobian to ndarray
            nalgebra_to_ndarray(&nalgebra_jacobian)
        }
        
        fn has_custom_jacobian(&self) -> bool {
            true  // LM problems always have a jacobian method
        }
    }
}

/// Adapter for using our `Problem` trait with the `levenberg-marquardt` crate.
///
/// This adapter allows problems defined using our `Problem` trait to be used with
/// the `levenberg-marquardt` crate's optimizer.
#[cfg(feature = "lm-compat")]
pub mod lm_adapter {
    use super::*;
    use levenberg_marquardt::{LeastSquaresProblem, LeastSquaresFunc};
    use nalgebra::{DMatrix, DVector};
    use crate::utils::matrix_convert::{ndarray_to_nalgebra, nalgebra_to_ndarray};
    
    /// Adapter to use our problem with the `levenberg-marquardt` crate.
    pub struct LmAdapter<P: Problem> {
        problem: P,
    }
    
    impl<P: Problem> LmAdapter<P> {
        /// Create a new adapter for our problem.
        pub fn new(problem: P) -> Self {
            Self { problem }
        }
    }
    
    impl<P: Problem> LeastSquaresProblem<f64> for LmAdapter<P> {
        type Params = DVector<f64>;
        type Residuals = DVector<f64>;
        type Jacobian = DMatrix<f64>;
        
        fn residuals(&self, params: &Self::Params) -> Result<Self::Residuals, levenberg_marquardt::Error> {
            // Convert nalgebra params to ndarray
            let ndarray_params = nalgebra_to_ndarray(params)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))?;
            
            // Call our problem's eval method
            let ndarray_residuals = self.problem.eval(&ndarray_params)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))?;
            
            // Convert ndarray residuals to nalgebra
            ndarray_to_nalgebra(&ndarray_residuals)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))
        }
        
        fn jacobian(&self, params: &Self::Params) -> Result<Self::Jacobian, levenberg_marquardt::Error> {
            // Convert nalgebra params to ndarray
            let ndarray_params = nalgebra_to_ndarray(params)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))?;
            
            // Call our problem's jacobian method
            let ndarray_jacobian = self.problem.jacobian(&ndarray_params)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))?;
            
            // Convert ndarray jacobian to nalgebra
            ndarray_to_nalgebra(&ndarray_jacobian)
                .map_err(|e| levenberg_marquardt::Error::InvalidInputs(format!("{}", e)))
        }
        
        fn params_len(&self) -> usize {
            self.problem.parameter_count()
        }
        
        fn residuals_len(&self) -> usize {
            self.problem.residual_count()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use approx::assert_relative_eq;
    
    /// A simple linear model for testing: f(x) = a * x + b
    struct LinearModel {
        x_data: Array1<f64>,
        y_data: Array1<f64>,
    }
    
    impl LinearModel {
        fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
            assert_eq!(x_data.len(), y_data.len(), "x and y data must have the same length");
            Self { x_data, y_data }
        }
    }
    
    impl Problem for LinearModel {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            if params.len() != 2 {
                return Err(LmOptError::DimensionMismatch(
                    format!("Expected 2 parameters, got {}", params.len())
                ));
            }
            
            let a = params[0];
            let b = params[1];
            
            let residuals = self.x_data.iter()
                .zip(self.y_data.iter())
                .map(|(x, y)| a * x + b - y)
                .collect::<Vec<f64>>();
            
            Ok(Array1::from_vec(residuals))
        }
        
        fn parameter_count(&self) -> usize {
            2  // a and b
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
    
    #[test]
    fn test_linear_model_eval() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2x
        let model = LinearModel::new(x, y);
        
        // Parameters [a, b] = [2, 0] should give zero residuals
        let params = array![2.0, 0.0];
        let residuals = model.eval(&params).unwrap();
        
        assert_eq!(residuals.len(), 5);
        for r in residuals.iter() {
            assert_relative_eq!(*r, 0.0, epsilon = 1e-10);
        }
        
        // Parameters [a, b] = [1, 0] should give residuals equal to -y/2
        let params = array![1.0, 0.0];
        let residuals = model.eval(&params).unwrap();
        
        assert_eq!(residuals.len(), 5);
        for (i, r) in residuals.iter().enumerate() {
            assert_relative_eq!(*r, -((i as f64) + 1.0), epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_linear_model_jacobian() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2x
        let model = LinearModel::new(x.clone(), y);
        
        let params = array![2.0, 0.0];
        let jacobian = model.jacobian(&params).unwrap();
        
        assert_eq!(jacobian.shape(), &[5, 2]);
        
        // First column should be x values
        for i in 0..5 {
            assert_eq!(jacobian[[i, 0]], x[i]);
        }
        
        // Second column should be all 1's
        for i in 0..5 {
            assert_eq!(jacobian[[i, 1]], 1.0);
        }
    }
    
    #[test]
    fn test_eval_cost() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];  // y = 2x
        let model = LinearModel::new(x, y);
        
        // Parameters [a, b] = [2, 0] should give zero cost
        let params = array![2.0, 0.0];
        let cost = model.eval_cost(&params).unwrap();
        assert_relative_eq!(cost, 0.0, epsilon = 1e-10);
        
        // Parameters [a, b] = [1, 0] should give cost = sum(i^2) for i in 1..=5
        let params = array![1.0, 0.0];
        let cost = model.eval_cost(&params).unwrap();
        let expected_cost = (1..=5).map(|i| (i as f64).powi(2)).sum::<f64>();
        assert_relative_eq!(cost, expected_cost, epsilon = 1e-10);
    }
}