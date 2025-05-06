//! Core Levenberg-Marquardt algorithm implementation.
//!
//! This module implements the Levenberg-Marquardt algorithm for nonlinear
//! least squares optimization, with trust region and adaptive damping.

use faer::{Mat, Col};
// Using our own solvers for now
use ndarray::{Array1, Array2};
use crate::error::{LmOptError, Result};
use crate::problem::Problem;
use crate::utils::matrix_convert::{ndarray_to_faer, faer_to_ndarray, ndarray_vec_to_faer, faer_vec_to_ndarray};

/// Configuration options for the Levenberg-Marquardt algorithm.
#[derive(Debug, Clone)]
pub struct LmConfig {
    /// Maximum number of iterations. Default: 100
    pub max_iterations: usize,
    
    /// Tolerance for change in residual norm. Default: 1e-8
    pub ftol: f64,
    
    /// Tolerance for change in parameter values. Default: 1e-8
    pub xtol: f64,
    
    /// Tolerance for gradient norm. Default: 1e-8
    pub gtol: f64,
    
    /// Initial value for the damping parameter. Default: 1e-3
    pub initial_lambda: f64,
    
    /// Factor by which to increase/decrease lambda. Default: 10.0
    pub lambda_factor: f64,
    
    /// Minimum value for lambda. Default: 1e-10
    pub min_lambda: f64,
    
    /// Maximum value for lambda. Default: 1e10
    pub max_lambda: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            initial_lambda: 1e-3,
            lambda_factor: 10.0,
            min_lambda: 1e-10,
            max_lambda: 1e10,
        }
    }
}

/// Result of the Levenberg-Marquardt optimization.
#[derive(Debug, Clone)]
pub struct LmResult {
    /// Optimized parameter values
    pub params: Array1<f64>,
    
    /// Residuals at the solution
    pub residuals: Array1<f64>,
    
    /// Sum of squared residuals
    pub cost: f64,
    
    /// Number of iterations performed
    pub iterations: usize,
    
    /// Whether the optimization succeeded
    pub success: bool,
    
    /// A message describing the result
    pub message: String,
    
    /// The Jacobian matrix at the solution (if requested)
    pub jacobian: Option<Array2<f64>>,
}

/// Status of the iteration.
enum IterationStatus {
    /// Continue iteration
    Continue,
    
    /// Converged successfully
    Converged(String),
    
    /// Failed to converge
    Failed(String),
}

/// The Levenberg-Marquardt optimizer.
#[derive(Debug, Clone)]
pub struct LevenbergMarquardt {
    /// Configuration options
    config: LmConfig,
}

impl LevenbergMarquardt {
    /// Create a new Levenberg-Marquardt optimizer with the given configuration.
    pub fn new(config: LmConfig) -> Self {
        Self { config }
    }
    
    /// Create a new Levenberg-Marquardt optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: LmConfig::default(),
        }
    }
    
    /// Minimize the sum of squared residuals for the given problem.
    ///
    /// This method implements the Levenberg-Marquardt algorithm to find the
    /// parameter values that minimize the sum of squared residuals.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem to solve
    /// * `initial_params` - Initial guess for the parameter values
    ///
    /// # Returns
    ///
    /// * `Result<LmResult>` - The result of the optimization
    pub fn minimize<P: Problem>(&self, problem: &P, initial_params: Array1<f64>) -> Result<LmResult> {
        // Check parameter dimensions
        let n_params = problem.parameter_count();
        if initial_params.len() != n_params {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}", n_params, initial_params.len()
            )));
        }
        
        // Initialize parameters and damping
        let mut params = initial_params;
        let mut lambda = self.config.initial_lambda;
        
        // Evaluate initial residuals and cost
        let mut residuals = problem.eval(&params)?;
        let mut cost = residuals.iter().map(|r| r.powi(2)).sum();
        
        // Iterate until convergence or max iterations
        let mut iterations = 0;
        let mut jacobian: Option<Array2<f64>> = None;
        
        loop {
            // Compute Jacobian
            jacobian = Some(problem.jacobian(&params)?);
            let j = ndarray_to_faer(jacobian.as_ref().unwrap())?;
            let r = ndarray_vec_to_faer(&residuals)?;
            
            // Compute gradient g = J^T * r
            let g = j.transpose() * &r;
            
            // Check gradient convergence
            let gradient_norm = g.norm_l2();
            if gradient_norm < self.config.gtol {
                return Ok(LmResult {
                    params,
                    residuals,
                    cost,
                    iterations,
                    success: true,
                    message: format!("Gradient convergence: ||g|| = {:.2e} < {:.2e}", gradient_norm, self.config.gtol),
                    jacobian,
                });
            }
            
            // Calculate step using trust region approach
            let step = match self.calculate_step(&j, &r, lambda)? {
                Some(s) => s,
                None => {
                    // If step calculation failed, increase lambda and try again
                    lambda = (lambda * self.config.lambda_factor).min(self.config.max_lambda);
                    if lambda == self.config.max_lambda {
                        return Err(LmOptError::ConvergenceFailure(
                            "Failed to calculate step, and lambda reached maximum".to_string()
                        ));
                    }
                    continue;
                }
            };
            
            // Convert step to ndarray
            let step_nd = faer_vec_to_ndarray(&step)?;
            
            // Update parameters
            let new_params = &params + &step_nd;
            
            // Evaluate new residuals and cost
            let new_residuals = problem.eval(&new_params)?;
            let new_cost = new_residuals.iter().map(|r| r.powi(2)).sum();
            
            // Check if the step reduced the cost
            if new_cost < cost {
                // Step accepted
                
                // Check convergence
                let param_change = (&new_params - &params).iter().map(|x| x.abs()).fold(0./0., f64::max);
                let cost_change = (cost - new_cost) / cost.max(1e-10);
                
                let status = if iterations >= self.config.max_iterations {
                    IterationStatus::Failed(format!(
                        "Maximum iterations ({}) reached", self.config.max_iterations
                    ))
                } else if param_change < self.config.xtol {
                    IterationStatus::Converged(format!(
                        "Parameter convergence: |dx|/|x| = {:.2e} < {:.2e}", param_change, self.config.xtol
                    ))
                } else if cost_change < self.config.ftol {
                    IterationStatus::Converged(format!(
                        "Cost convergence: |df|/|f| = {:.2e} < {:.2e}", cost_change, self.config.ftol
                    ))
                } else {
                    IterationStatus::Continue
                };
                
                // Update state
                params = new_params;
                residuals = new_residuals;
                cost = new_cost;
                lambda = (lambda / self.config.lambda_factor).max(self.config.min_lambda);
                iterations += 1;
                
                // Handle convergence or failure
                match status {
                    IterationStatus::Continue => (),
                    IterationStatus::Converged(message) => {
                        return Ok(LmResult {
                            params,
                            residuals,
                            cost,
                            iterations,
                            success: true,
                            message,
                            jacobian: None,  // We'll recompute this at the end
                        });
                    }
                    IterationStatus::Failed(message) => {
                        return Ok(LmResult {
                            params,
                            residuals,
                            cost,
                            iterations,
                            success: false,
                            message,
                            jacobian: None,  // We'll recompute this at the end
                        });
                    }
                }
            } else {
                // Step rejected - increase lambda and try again
                lambda = (lambda * self.config.lambda_factor).min(self.config.max_lambda);
                
                if lambda == self.config.max_lambda {
                    return Ok(LmResult {
                        params,
                        residuals,
                        cost,
                        iterations,
                        success: false,
                        message: "Failed to decrease cost, and lambda reached maximum".to_string(),
                        jacobian: None,  // We'll recompute this at the end
                    });
                }
            }
        }
    }
    
    /// Calculate the Levenberg-Marquardt step.
    ///
    /// This method solves the equation (J^T J + λI) δ = J^T r, where:
    /// - J is the Jacobian matrix
    /// - r is the residual vector
    /// - λ is the damping parameter
    /// - δ is the step
    ///
    /// # Arguments
    ///
    /// * `j` - The Jacobian matrix
    /// * `r` - The residual vector
    /// * `lambda` - The damping parameter
    ///
    /// # Returns
    ///
    /// * `Result<Option<Col<f64>>>` - The step, or None if the system is singular
    fn calculate_step(&self, j: &Mat<f64>, r: &Col<f64>, lambda: f64) -> Result<Option<Col<f64>>> {
        // Convert to ndarray for simpler operations
        let j_ndarray = faer_to_ndarray(j)?;
        let r_ndarray = faer_vec_to_ndarray(r)?;
        
        // Compute J^T (transpose of J)
        let jt_ndarray = j_ndarray.t().to_owned();
        
        // Compute normal equations matrix: J^T J + lambda * I
        let jtj_ndarray = jt_ndarray.dot(&j_ndarray);
        
        let n = jtj_ndarray.nrows();
        let mut a_ndarray = jtj_ndarray.clone();
        
        // Add damping to diagonal
        for i in 0..n {
            a_ndarray[[i, i]] += lambda;
        }
        
        // Compute J^T r
        let jtr_ndarray = jt_ndarray.dot(&r_ndarray);
        
        // Try to solve the system using a simple Cholesky decomposition
        let mut a_mutable = a_ndarray.clone();
        
        // Simple Cholesky decomposition (without pivoting)
        let mut cholesky_success = true;
        for k in 0..n {
            // Diagonal element
            for j in 0..k {
                a_mutable[[k, k]] -= a_mutable[[k, j]] * a_mutable[[k, j]];
            }
            
            if a_mutable[[k, k]] <= 0.0 {
                cholesky_success = false;
                break;
            }
            
            let akk_sqrt = a_mutable[[k, k]].sqrt();
            a_mutable[[k, k]] = akk_sqrt;
            
            // Update column elements
            for i in k+1..n {
                for j in 0..k {
                    a_mutable[[i, k]] -= a_mutable[[i, j]] * a_mutable[[k, j]];
                }
                a_mutable[[i, k]] /= akk_sqrt;
            }
        }
        
        if !cholesky_success {
            // Matrix is not positive definite, use simple QR decomposition
            // For simplicity, we'll use the normal equations: J^T J dx = -J^T r
            // where r is the residual vector
            
            // Solve using Gauss-Seidel iteration
            let mut step_ndarray = Array1::zeros(n);
            
            // Now refine using Gauss-Seidel iterations
            let max_iter = 10;  // Maximum number of iterations
            let tol = 1e-8;     // Tolerance for convergence
            
            for _ in 0..max_iter {
                let mut max_change: f64 = 0.0;
                
                for i in 0..n {
                    let mut sum = 0.0;
                    
                    // Sum contributions from other variables (updated and not yet updated)
                    for j in 0..n {
                        if i != j {
                            sum += a_ndarray[[i, j]] * step_ndarray[j];
                        }
                    }
                    
                    // Update the variable
                    let old_val: f64 = step_ndarray[i];
                    let new_val: f64 = (jtr_ndarray[i] - sum) / a_ndarray[[i, i]];
                    let change: f64 = (new_val - old_val).abs();
                    step_ndarray[i] = new_val;
                    
                    // Track maximum change
                    max_change = f64::max(max_change, change);
                }
                
                // Check for convergence
                if max_change < tol {
                    break;
                }
            }
            
            // Convert back to faer
            let step = ndarray_vec_to_faer(&(-step_ndarray))?;
            Ok(Some(step))
        } else {
            // Cholesky succeeded, so use it to solve the system
            let mut step_ndarray = Array1::zeros(n);
            let mut rhs = jtr_ndarray.clone();
            
            // Forward substitution (L * y = b)
            for i in 0..n {
                for j in 0..i {
                    rhs[i] -= a_mutable[[i, j]] * rhs[j];
                }
                rhs[i] /= a_mutable[[i, i]];
            }
            
            // Backward substitution (L^T * x = y)
            for i in (0..n).rev() {
                step_ndarray[i] = rhs[i];
                for j in (i+1)..n {
                    step_ndarray[i] -= a_mutable[[j, i]] * step_ndarray[j];
                }
                step_ndarray[i] /= a_mutable[[i, i]];
            }
            
            // Convert back to faer
            let step = ndarray_vec_to_faer(&(-step_ndarray))?;
            Ok(Some(step))
        }
    }
}

/// Module containing utility functions for the LM algorithm.
pub mod utils {
    use super::*;
    
    /// Check if the algorithm has converged based on various criteria.
    pub fn check_convergence(
        params: &Array1<f64>,
        new_params: &Array1<f64>,
        cost: f64,
        new_cost: f64,
        gradient_norm: f64,
        config: &LmConfig,
    ) -> Option<String> {
        // Check parameter change
        let param_change_vec: Vec<f64> = new_params.iter().zip(params.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let param_change = param_change_vec.iter().fold(0./0., |a, &b| f64::max(a, b));
        if param_change < config.xtol {
            return Some(format!("Parameter convergence: |dx|/|x| = {:.2e} < {:.2e}", param_change, config.xtol));
        }
        
        // Check cost change
        let cost_change = (cost - new_cost) / cost.max(1e-10);
        if cost_change < config.ftol {
            return Some(format!("Cost convergence: |df|/|f| = {:.2e} < {:.2e}", cost_change, config.ftol));
        }
        
        // Check gradient norm
        if gradient_norm < config.gtol {
            return Some(format!("Gradient convergence: ||g|| = {:.2e} < {:.2e}", gradient_norm, config.gtol));
        }
        
        // Not converged
        None
    }
    
    /// Update the damping parameter based on the success of the step.
    pub fn update_lambda(
        lambda: f64,
        success: bool,
        config: &LmConfig,
    ) -> f64 {
        if success {
            // Step accepted - decrease lambda
            (lambda / config.lambda_factor).max(config.min_lambda)
        } else {
            // Step rejected - increase lambda
            (lambda * config.lambda_factor).min(config.max_lambda)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use approx::assert_relative_eq;
    use crate::problem::Problem;
    
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
    fn test_linear_fit() {
        // Create test data: y = 2x + 3 + noise
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.1, 7.0, 8.9, 11.2, 13.0];  // Approximately 2x + 3
        
        let model = LinearModel::new(x, y);
        let initial_params = array![1.0, 1.0];  // Initial guess [a, b] = [1, 1]
        
        let lm = LevenbergMarquardt::with_default_config();
        let result = lm.minimize(&model, initial_params).unwrap();
        
        // Check that the fit succeeded
        assert!(result.success);
        
        // Check that the parameters are close to [2, 3]
        assert_relative_eq!(result.params[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(result.params[1], 3.0, epsilon = 0.1);
        
        // Check that the cost is small
        assert!(result.cost < 0.1);
    }
    
    #[test]
    fn test_quadratic_model() {
        // Create a new model for a quadratic function
        struct QuadraticModel {
            x_data: Array1<f64>,
            y_data: Array1<f64>,
        }
        
        impl QuadraticModel {
            fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
                assert_eq!(x_data.len(), y_data.len(), "x and y data must have the same length");
                Self { x_data, y_data }
            }
        }
        
        impl Problem for QuadraticModel {
            fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
                if params.len() != 3 {
                    return Err(LmOptError::DimensionMismatch(
                        format!("Expected 3 parameters, got {}", params.len())
                    ));
                }
                
                let a = params[0];
                let b = params[1];
                let c = params[2];
                
                let residuals = self.x_data.iter()
                    .zip(self.y_data.iter())
                    .map(|(x, y)| a * x.powi(2) + b * x + c - y)
                    .collect::<Vec<f64>>();
                
                Ok(Array1::from_vec(residuals))
            }
            
            fn parameter_count(&self) -> usize {
                3  // a, b, c
            }
            
            fn residual_count(&self) -> usize {
                self.x_data.len()
            }
            
            fn jacobian(&self, _params: &Array1<f64>) -> Result<Array2<f64>> {
                let n = self.x_data.len();
                let mut jac = Array2::zeros((n, 3));
                
                for i in 0..n {
                    let x = self.x_data[i];
                    // Derivative with respect to a
                    jac[[i, 0]] = x.powi(2);
                    // Derivative with respect to b
                    jac[[i, 1]] = x;
                    // Derivative with respect to c
                    jac[[i, 2]] = 1.0;
                }
                
                Ok(jac)
            }
            
            fn has_custom_jacobian(&self) -> bool {
                true
            }
        }
        
        // Create test data: y = 2x^2 - 3x + 1 + noise
        let x = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = array![11.8, 5.9, 1.1, 0.1, 3.0, 9.9];  // Approximately 2x^2 - 3x + 1
        
        let model = QuadraticModel::new(x, y);
        let initial_params = array![1.0, 1.0, 1.0];  // Initial guess [a, b, c] = [1, 1, 1]
        
        let lm = LevenbergMarquardt::with_default_config();
        let result = lm.minimize(&model, initial_params).unwrap();
        
        // Check that the fit succeeded
        assert!(result.success);
        
        // Check that the parameters are close to [2, -3, 1]
        assert_relative_eq!(result.params[0], 2.0, epsilon = 0.1);
        assert_relative_eq!(result.params[1], -3.0, epsilon = 0.1);
        assert_relative_eq!(result.params[2], 1.0, epsilon = 0.1);
        
        // Check that the cost is small
        assert!(result.cost < 1.0);
    }
}