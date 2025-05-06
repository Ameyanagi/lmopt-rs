//! Integration tests for the Levenberg-Marquardt algorithm.

use ndarray::{array, Array1, Array2};
use approx::assert_relative_eq;
use lmopt_rs::{Problem, LmOptError, Result};
use lmopt_rs::lm::{LevenbergMarquardt, LmConfig};

/// Test Problem: Simple 1D linear function f(x) = a*x + b
struct LinearProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl LinearProblem {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        assert_eq!(x_data.len(), y_data.len(), "Data dimensions must match");
        Self { x_data, y_data }
    }
}

impl Problem for LinearProblem {
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
        2
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
    
    fn jacobian(&self, _params: &Array1<f64>) -> Result<Array2<f64>> {
        let n = self.x_data.len();
        let mut jac = Array2::zeros((n, 2));
        
        for i in 0..n {
            jac[[i, 0]] = self.x_data[i];  // d/da
            jac[[i, 1]] = 1.0;  // d/db
        }
        
        Ok(jac)
    }
    
    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

/// Test Problem: Rosenbrock function
struct RosenbrockProblem;

impl Problem for RosenbrockProblem {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 2 {
            return Err(LmOptError::DimensionMismatch(
                format!("Expected 2 parameters, got {}", params.len())
            ));
        }
        
        let x = params[0];
        let y = params[1];
        
        // For least squares, we return the residuals as a vector
        // The Rosenbrock function f(x,y) = (1-x)² + 100(y-x²)²
        // can be written as the sum of squares of:
        //   r₁ = 1 - x
        //   r₂ = 10(y - x²)
        
        let r1 = 1.0 - x;
        let r2 = 10.0 * (y - x.powi(2));
        
        Ok(array![r1, r2])
    }
    
    fn parameter_count(&self) -> usize {
        2
    }
    
    fn residual_count(&self) -> usize {
        2
    }
    
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        let x = params[0];
        
        // Jacobian matrix:
        // [ dr₁/dx, dr₁/dy ]   [ -1,  0      ]
        // [ dr₂/dx, dr₂/dy ] = [ -20x, 10    ]
        
        let mut jac = Array2::zeros((2, 2));
        jac[[0, 0]] = -1.0;
        jac[[0, 1]] = 0.0;
        jac[[1, 0]] = -20.0 * x;
        jac[[1, 1]] = 10.0;
        
        Ok(jac)
    }
    
    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

/// Test Problem: Exponential function fit
struct ExponentialProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl ExponentialProblem {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        assert_eq!(x_data.len(), y_data.len(), "Data dimensions must match");
        Self { x_data, y_data }
    }
}

impl Problem for ExponentialProblem {
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
            .map(|(x, y)| a * (-b * x).exp() - y)
            .collect::<Vec<f64>>();
        
        Ok(Array1::from_vec(residuals))
    }
    
    fn parameter_count(&self) -> usize {
        2
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
    
    // Jacobian is provided by the default implementation using finite differences
}

#[test]
fn test_linear_fitting() {
    // Create test data: y = 3x + 2 + noise
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.1, 4.9, 8.05, 10.8, 14.1, 17.0];
    
    let problem = LinearProblem::new(x, y);
    
    // Initial parameter guess [a, b] = [1, 1]
    let initial_params = array![1.0, 1.0];
    
    // Create optimizer
    let lm = LevenbergMarquardt::with_default_config();
    
    // Run optimization
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // Check that the optimization succeeded
    assert!(result.success);
    
    // Check that the parameters are close to the expected values
    assert_relative_eq!(result.params[0], 3.0, epsilon = 0.1);  // a ≈ 3
    assert_relative_eq!(result.params[1], 2.0, epsilon = 0.1);  // b ≈ 2
    
    // Check that the cost function is small
    assert!(result.cost < 0.1);
}

#[test]
fn test_rosenbrock_optimization() {
    // The Rosenbrock function has a minimum at (1, 1)
    let problem = RosenbrockProblem;
    
    // Start from a point far from the minimum
    let initial_params = array![-1.2, 1.0];
    
    // Create optimizer with tighter convergence criteria
    let config = LmConfig {
        max_iterations: 200,
        ftol: 1e-10,
        xtol: 1e-10,
        gtol: 1e-10,
        ..LmConfig::default()
    };
    
    let lm = LevenbergMarquardt::new(config);
    
    // Run optimization
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // Check that the optimization succeeded
    assert!(result.success);
    
    // Check that the parameters are close to the minimum at (1, 1)
    assert_relative_eq!(result.params[0], 1.0, epsilon = 1e-4);
    assert_relative_eq!(result.params[1], 1.0, epsilon = 1e-4);
    
    // Check that the cost function is close to zero
    assert!(result.cost < 1e-8);
}

#[test]
fn test_exponential_fitting() {
    // Create test data: y = 2 * exp(-0.5 * x) + noise
    let x = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let y = array![
        2.02, 1.67, 1.21, 0.98, 0.81, 0.62, 0.45, 0.39, 0.29
    ];
    
    let problem = ExponentialProblem::new(x, y);
    
    // Initial parameter guess [a, b] = [1, 0.1]
    let initial_params = array![1.0, 0.1];
    
    // Create optimizer
    let lm = LevenbergMarquardt::with_default_config();
    
    // Run optimization
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // Check that the optimization succeeded
    assert!(result.success);
    
    // Check that the parameters are close to the expected values
    assert_relative_eq!(result.params[0], 2.0, epsilon = 0.1);  // a ≈ 2
    assert_relative_eq!(result.params[1], 0.5, epsilon = 0.1);  // b ≈ 0.5
    
    // Check that the cost function is small
    assert!(result.cost < 0.01);
}

#[test]
fn test_bad_initial_guess() {
    // Create test data: y = 3x + 2
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.0, 5.0, 8.0, 11.0, 14.0, 17.0];
    
    let problem = LinearProblem::new(x, y);
    
    // Very poor initial guess [a, b] = [100, -50]
    let initial_params = array![100.0, -50.0];
    
    // Create optimizer
    let lm = LevenbergMarquardt::with_default_config();
    
    // Run optimization
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // Check that the optimization succeeded despite bad initial guess
    assert!(result.success);
    
    // Check that the parameters are close to the expected values
    assert_relative_eq!(result.params[0], 3.0, epsilon = 0.1);  // a ≈ 3
    assert_relative_eq!(result.params[1], 2.0, epsilon = 0.1);  // b ≈ 2
}

#[test]
fn test_parameter_bounds() {
    // TODO: Implement parameter bounds in Phase 2
    // This test is a placeholder for future development
}

#[test]
fn test_custom_config() {
    // Create a simple problem
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
    
    let problem = LinearProblem::new(x, y);
    let initial_params = array![1.0, 0.0];
    
    // Create optimizer with custom configuration
    let config = LmConfig {
        max_iterations: 5,  // Very few iterations
        ftol: 1e-2,         // Loose tolerance
        xtol: 1e-2,
        gtol: 1e-2,
        initial_lambda: 1.0, // Different initial lambda
        ..LmConfig::default()
    };
    
    let lm = LevenbergMarquardt::new(config);
    
    // Run optimization
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // The problem is simple enough that it should converge even with few iterations
    assert!(result.success);
    
    // Check that iterations is small
    assert!(result.iterations <= 5);
    
    // Check that the parameters are still reasonable
    assert_relative_eq!(result.params[0], 2.0, epsilon = 0.2);
    assert_relative_eq!(result.params[1], 1.0, epsilon = 0.2);
}