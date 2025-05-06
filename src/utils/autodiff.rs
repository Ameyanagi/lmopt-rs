//! Automatic differentiation utilities for the lmopt-rs library.
//!
//! This module provides functions for computing derivatives and Jacobians
//! using automatic differentiation with Rust's nightly `std::autodiff` feature.
//! When autodiff is not available, it falls back to numerical differentiation.

use ndarray::{Array1, Array2};
use crate::error::{LmOptError, Result};
use crate::problem::Problem;
use crate::utils::finite_difference;

// Flag to track if autodiff is available
// This is determined at compile time based on the nightly feature
#[allow(dead_code)]
const HAS_AUTODIFF: bool = cfg!(feature = "autodiff");

/// Compute the Jacobian matrix using automatic differentiation.
///
/// The Jacobian is the matrix of partial derivatives of the residuals with
/// respect to the parameters: J[i,j] = ∂residual[i]/∂param[j].
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `params` - The parameter values at which to evaluate the Jacobian
///
/// # Returns
///
/// * `Result<Array2<f64>>` - The Jacobian matrix
///
/// # Notes
///
/// If autodiff is available (using the nightly feature), it will be used.
/// Otherwise, this function falls back to numerical differentiation using
/// finite differences.
pub fn jacobian<P: Problem>(problem: &P, params: &Array1<f64>) -> Result<Array2<f64>> {
    // If the problem has a custom jacobian, use it
    if problem.has_custom_jacobian() {
        return problem.jacobian(params);
    }
    
    // Otherwise use finite differences (autodiff will be implemented later)
    finite_difference::jacobian(problem, params, None)
}

/// Compute the gradient of a scalar function using automatic differentiation.
///
/// The gradient is the vector of partial derivatives of the function with
/// respect to the parameters: grad[j] = ∂f/∂param[j].
///
/// # Arguments
///
/// * `f` - The function to differentiate
/// * `params` - The parameter values at which to evaluate the gradient
///
/// # Returns
///
/// * `Result<Array1<f64>>` - The gradient vector
///
/// # Notes
///
/// If autodiff is available (using the nightly feature), it will be used.
/// Otherwise, this function falls back to numerical differentiation using
/// finite differences.
pub fn gradient<F>(f: F, params: &Array1<f64>) -> Result<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64>
{
    // Use finite differences (autodiff will be implemented later)
    finite_difference::gradient(f, params, None)
}

/// Compute the Hessian matrix using automatic differentiation.
///
/// The Hessian is the matrix of second partial derivatives of the function with
/// respect to the parameters: H[i,j] = ∂²f/∂param[i]∂param[j].
///
/// # Arguments
///
/// * `f` - The function to differentiate
/// * `params` - The parameter values at which to evaluate the Hessian
///
/// # Returns
///
/// * `Result<Array2<f64>>` - The Hessian matrix
///
/// # Notes
///
/// If autodiff is available (using the nightly feature), it will be used.
/// Otherwise, this function falls back to numerical differentiation using
/// finite differences.
pub fn hessian<F>(f: F, params: &Array1<f64>) -> Result<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64>
{
    // Use finite differences (autodiff will be implemented later)
    finite_difference::hessian(f, params, None)
}

// The following block will be enabled when the autodiff feature is available
#[cfg(feature = "autodiff")]
mod autodiff_impl {
    use super::*;
    use std::autodiff::Forward;
    
    // Implementations using std::autodiff will go here
    // This is a placeholder for future implementation
}

// When the appropriate autodiff API is stabilized, we can add real implementations here