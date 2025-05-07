//! Configuration options for the Levenberg-Marquardt algorithm.
//!
//! This module defines the configuration options and parameter settings for the
//! Levenberg-Marquardt algorithm, including convergence criteria, step calculation
//! methods, and robust fitting options.

use super::robust::RobustLoss;

/// Method for calculating the Jacobian matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiffMethod {
    /// Use finite differences to approximate the Jacobian
    FiniteDifference,

    /// Use automatic differentiation to calculate the Jacobian
    #[cfg(feature = "autodiff")]
    AutoDiff,

    /// Use the analytical Jacobian provided by the problem implementation
    Analytical,
}

impl Default for DiffMethod {
    fn default() -> Self {
        DiffMethod::FiniteDifference
    }
}

/// Method for solving the linear system in the Levenberg-Marquardt step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecompositionMethod {
    /// Use Cholesky decomposition (fastest but requires positive definite matrix)
    Cholesky,

    /// Use QR decomposition (more stable than Cholesky, works for non-square matrices)
    QR,

    /// Use SVD decomposition (most stable but slowest, handles rank-deficient matrices)
    SVD,

    /// Automatically select the best method based on the problem
    Auto,
}

impl Default for DecompositionMethod {
    fn default() -> Self {
        DecompositionMethod::Auto
    }
}

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

    /// Factor by which to increase lambda. Default: 10.0
    pub lambda_up_factor: f64,

    /// Factor by which to decrease lambda. Default: 0.1
    pub lambda_down_factor: f64,

    /// Minimum value for lambda. Default: 1e-10
    pub min_lambda: f64,

    /// Maximum value for lambda. Default: 1e10
    pub max_lambda: f64,

    /// Method to use for calculating the Jacobian. Default: FiniteDifference
    pub diff_method: DiffMethod,

    /// Method to use for solving the linear system. Default: Auto
    pub decomposition_method: DecompositionMethod,

    /// Whether to calculate and return the Jacobian at the solution. Default: false
    pub calc_jacobian: bool,

    /// The robust loss function to use for outlier-resistant fitting. Default: LeastSquares
    pub loss_function: RobustLoss,

    /// Number of iterations of iteratively reweighted least squares (IRLS) for robust fitting. Default: 1
    pub robust_iterations: usize,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            initial_lambda: 1e-3,
            lambda_up_factor: 10.0,
            lambda_down_factor: 0.1,
            min_lambda: 1e-10,
            max_lambda: 1e10,
            diff_method: DiffMethod::default(),
            decomposition_method: DecompositionMethod::default(),
            calc_jacobian: false,
            loss_function: RobustLoss::default(),
            robust_iterations: 1,
        }
    }
}
