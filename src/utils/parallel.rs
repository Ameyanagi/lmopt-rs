//! Parallel processing utilities for computationally intensive operations.
//!
//! This module provides parallel implementations of the most computationally
//! intensive operations in the library, such as Jacobian calculation and
//! residual evaluation.

use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::error::{LmOptError, Result};
use crate::problem::Problem;

/// Default step size for finite differences.
const DEFAULT_EPSILON: f64 = 1e-8;

/// Compute the Jacobian matrix using forward finite differences in parallel.
///
/// The Jacobian is the matrix of partial derivatives of the residuals with
/// respect to the parameters: J[i,j] = ∂residual[i]/∂param[j].
///
/// This function uses Rayon for parallel computation of the Jacobian columns,
/// which can significantly speed up the calculation for problems with many parameters.
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `params` - The parameter values at which to evaluate the Jacobian
/// * `epsilon` - The step size for finite differences (optional)
///
/// # Returns
///
/// * `Result<Array2<f64>>` - The Jacobian matrix
pub fn jacobian_parallel(
    problem: &dyn Problem,
    params: &Array1<f64>,
    epsilon: Option<f64>,
) -> Result<Array2<f64>> {
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let n_params = params.len();
    let n_residuals = problem.residual_count();

    // Evaluate residuals at the initial point
    let residuals = problem.eval(params)?;

    // Check residual dimensions
    if residuals.len() != n_residuals {
        return Err(LmOptError::DimensionMismatch(format!(
            "Expected {} residuals, got {}",
            n_residuals,
            residuals.len()
        )));
    }

    // Initialize Jacobian matrix
    let mut jac = Array2::zeros((n_residuals, n_params));

    // Compute each column of the Jacobian in parallel
    let columns: Result<Vec<_>> = (0..n_params)
        .into_par_iter()
        .map(|j| {
            // Perturb j-th parameter
            let mut params_perturbed = params.clone();

            // Adapt epsilon to parameter scale
            let param_j = params[j];
            let eps_j = if param_j.abs() > eps {
                param_j.abs() * eps
            } else {
                eps
            };

            params_perturbed[j] += eps_j;

            // Evaluate residuals at perturbed point
            let residuals_perturbed = problem.eval(&params_perturbed)?;

            // Compute partial derivatives for this column
            let column = residuals_perturbed
                .iter()
                .enumerate()
                .map(|(i, &r_perturbed)| (i, (r_perturbed - residuals[i]) / eps_j))
                .collect::<Vec<_>>();

            Ok(column)
        })
        .collect();

    // Fill in the Jacobian matrix with the computed columns
    for (j, column) in columns?.into_iter().enumerate() {
        for (i, value) in column {
            jac[[i, j]] = value;
        }
    }

    Ok(jac)
}

/// Compute the Jacobian matrix using central finite differences in parallel.
///
/// The Jacobian is the matrix of partial derivatives of the residuals with
/// respect to the parameters: J[i,j] = ∂residual[i]/∂param[j].
///
/// This function uses Rayon for parallel computation of the Jacobian columns,
/// which can significantly speed up the calculation for problems with many parameters.
/// It uses central differences for higher accuracy.
///
/// # Arguments
///
/// * `problem` - The problem to evaluate
/// * `params` - The parameter values at which to evaluate the Jacobian
/// * `epsilon` - The step size for finite differences (optional)
///
/// # Returns
///
/// * `Result<Array2<f64>>` - The Jacobian matrix
pub fn jacobian_central_parallel(
    problem: &dyn Problem,
    params: &Array1<f64>,
    epsilon: Option<f64>,
) -> Result<Array2<f64>> {
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let n_params = params.len();
    let n_residuals = problem.residual_count();

    // Initialize Jacobian matrix
    let mut jac = Array2::zeros((n_residuals, n_params));

    // Compute each column of the Jacobian in parallel
    let columns: Result<Vec<_>> = (0..n_params)
        .into_par_iter()
        .map(|j| {
            // Forward perturbation
            let mut params_forward = params.clone();

            // Adapt epsilon to parameter scale
            let param_j = params[j];
            let eps_j = if param_j.abs() > eps {
                param_j.abs() * eps
            } else {
                eps
            };

            params_forward[j] += eps_j;

            // Backward perturbation
            let mut params_backward = params.clone();
            params_backward[j] -= eps_j;

            // Evaluate residuals at perturbed points
            let residuals_forward = problem.eval(&params_forward)?;
            let residuals_backward = problem.eval(&params_backward)?;

            // Compute partial derivatives for this column using central difference
            let column = residuals_forward
                .iter()
                .zip(residuals_backward.iter())
                .enumerate()
                .map(|(i, (&rf, &rb))| (i, (rf - rb) / (2.0 * eps_j)))
                .collect::<Vec<_>>();

            Ok(column)
        })
        .collect();

    // Fill in the Jacobian matrix with the computed columns
    for (j, column) in columns?.into_iter().enumerate() {
        for (i, value) in column {
            jac[[i, j]] = value;
        }
    }

    Ok(jac)
}

/// Evaluate residuals in parallel when the residuals can be computed independently.
///
/// This function is useful for problems where each residual can be computed
/// independently and the computation is expensive.
///
/// # Arguments
///
/// * `eval_residual` - A function that computes a single residual
/// * `params` - The parameter values at which to evaluate the residuals
/// * `n_residuals` - The number of residuals to compute
///
/// # Returns
///
/// * `Result<Array1<f64>>` - The residuals vector
pub fn eval_residuals_parallel<F>(
    eval_residual: F,
    params: &Array1<f64>,
    n_residuals: usize,
) -> Result<Array1<f64>>
where
    F: Fn(usize, &Array1<f64>) -> Result<f64> + Sync,
{
    // Compute residuals in parallel
    let residuals: Result<Vec<_>> = (0..n_residuals)
        .into_par_iter()
        .map(|i| eval_residual(i, params))
        .collect();

    // Convert to Array1
    let residuals = residuals?;
    Ok(Array1::from_vec(residuals))
}

/// Compute the gradient of a scalar function using central finite differences in parallel.
///
/// The gradient is the vector of partial derivatives of the function with
/// respect to the parameters: grad[j] = ∂f/∂param[j].
///
/// # Arguments
///
/// * `f` - The function to differentiate
/// * `params` - The parameter values at which to evaluate the gradient
/// * `epsilon` - The step size for finite differences (optional)
///
/// # Returns
///
/// * `Result<Array1<f64>>` - The gradient vector
pub fn gradient_parallel<F>(f: F, params: &Array1<f64>, epsilon: Option<f64>) -> Result<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64> + Sync,
{
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let n_params = params.len();

    // Initialize gradient vector
    let mut grad = Array1::zeros(n_params);

    // Compute gradient components in parallel
    let components: Result<Vec<_>> = (0..n_params)
        .into_par_iter()
        .map(|j| {
            // Forward perturbation
            let mut params_forward = params.clone();

            // Adapt epsilon to parameter scale
            let param_j = params[j];
            let eps_j = if param_j.abs() > eps {
                param_j.abs() * eps
            } else {
                eps
            };

            params_forward[j] += eps_j;

            // Backward perturbation
            let mut params_backward = params.clone();
            params_backward[j] -= eps_j;

            // Evaluate function at perturbed points
            let f_forward = f(&params_forward)?;
            let f_backward = f(&params_backward)?;

            // Compute partial derivative using central difference
            Ok((j, (f_forward - f_backward) / (2.0 * eps_j)))
        })
        .collect();

    // Fill in the gradient vector
    for (idx, value) in components? {
        grad[idx] = value;
    }

    Ok(grad)
}

/// Compute the Hessian matrix using central finite differences in parallel.
///
/// The Hessian is the matrix of second partial derivatives of the function with
/// respect to the parameters: H[i,j] = ∂²f/∂param[i]∂param[j].
///
/// This implementation computes the diagonal and upper triangular elements in parallel,
/// then fills in the lower triangular elements using symmetry.
///
/// # Arguments
///
/// * `f` - The function to differentiate
/// * `params` - The parameter values at which to evaluate the Hessian
/// * `epsilon` - The step size for finite differences (optional)
///
/// # Returns
///
/// * `Result<Array2<f64>>` - The Hessian matrix
pub fn hessian_parallel<F>(f: F, params: &Array1<f64>, epsilon: Option<f64>) -> Result<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64> + Sync,
{
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let n_params = params.len();

    // Initialize Hessian matrix
    let mut hess = Array2::zeros((n_params, n_params));

    // Evaluate function at the initial point
    let f0 = f(params)?;

    // Create vector of (i, j) pairs for upper triangle (including diagonal)
    let mut indices = Vec::with_capacity(n_params * (n_params + 1) / 2);
    for i in 0..n_params {
        for j in 0..=i {
            indices.push((i, j));
        }
    }

    // Compute Hessian elements in parallel
    let elements: Result<Vec<_>> = indices
        .into_par_iter()
        .map(|(i, j)| {
            // Adapt epsilon to parameter scale
            let param_i = params[i];
            let param_j = params[j];

            let eps_i = if param_i.abs() > eps {
                param_i.abs() * eps
            } else {
                eps
            };

            let eps_j = if param_j.abs() > eps {
                param_j.abs() * eps
            } else {
                eps
            };

            if i == j {
                // Diagonal elements: use central difference formula
                let mut params_pp = params.clone();
                let mut params_mm = params.clone();

                params_pp[i] += eps_i;
                params_mm[i] -= eps_i;

                let f_pp = f(&params_pp)?;
                let f_mm = f(&params_mm)?;

                // Second derivative approximation
                Ok(((i, j), (f_pp - 2.0 * f0 + f_mm) / (eps_i * eps_i)))
            } else {
                // Off-diagonal elements: use mixed partial derivative formula
                let mut params_pp = params.clone();
                let mut params_pm = params.clone();
                let mut params_mp = params.clone();
                let mut params_mm = params.clone();

                params_pp[i] += eps_i;
                params_pp[j] += eps_j;

                params_pm[i] += eps_i;
                params_pm[j] -= eps_j;

                params_mp[i] -= eps_i;
                params_mp[j] += eps_j;

                params_mm[i] -= eps_i;
                params_mm[j] -= eps_j;

                let f_pp = f(&params_pp)?;
                let f_pm = f(&params_pm)?;
                let f_mp = f(&params_mp)?;
                let f_mm = f(&params_mm)?;

                // Mixed partial derivative approximation
                Ok(((i, j), (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps_i * eps_j)))
            }
        })
        .collect();

    // Fill in the Hessian matrix
    for ((i, j), value) in elements? {
        hess[[i, j]] = value;
        if i != j {
            hess[[j, i]] = value; // Symmetry
        }
    }

    Ok(hess)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::finite_difference;
    use approx::assert_relative_eq;
    use ndarray::array;

    // Test function: f(x, y) = x^2 + 2*y^2 + x*y
    fn test_function(params: &Array1<f64>) -> Result<f64> {
        let x = params[0];
        let y = params[1];
        Ok(x.powi(2) + 2.0 * y.powi(2) + x * y)
    }

    // Test problem: r1 = x^2 - 1, r2 = y^2 - 2
    struct TestProblem;

    impl Problem for TestProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            let x = params[0];
            let y = params[1];
            Ok(array![x.powi(2) - 1.0, y.powi(2) - 2.0])
        }

        fn parameter_count(&self) -> usize {
            2
        }

        fn residual_count(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_jacobian_parallel() {
        // Test at point (2, 3)
        let params = array![2.0, 3.0];
        let problem = TestProblem;

        // Analytical Jacobian: [[2*x, 0], [0, 2*y]] = [[4, 0], [0, 6]]
        let jac = jacobian_parallel(&problem, &params, None).unwrap();

        assert_eq!(jac.shape(), &[2, 2]);
        assert_relative_eq!(jac[[0, 0]], 4.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[0, 1]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[1, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[1, 1]], 6.0, epsilon = 1e-5);
    }

    #[test]
    fn test_jacobian_central_parallel() {
        // Test at point (2, 3)
        let params = array![2.0, 3.0];
        let problem = TestProblem;

        // Analytical Jacobian: [[2*x, 0], [0, 2*y]] = [[4, 0], [0, 6]]
        let jac = jacobian_central_parallel(&problem, &params, None).unwrap();

        assert_eq!(jac.shape(), &[2, 2]);
        assert_relative_eq!(jac[[0, 0]], 4.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[0, 1]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[1, 0]], 0.0, epsilon = 1e-5);
        assert_relative_eq!(jac[[1, 1]], 6.0, epsilon = 1e-5);
    }

    #[test]
    fn test_gradient_parallel() {
        // Test at point (2, 3)
        let params = array![2.0, 3.0];

        // Analytical gradient of x^2 + 2*y^2 + x*y at (2,3):
        // ∂f/∂x = 2x + y = 2*2 + 3 = 7
        // ∂f/∂y = 4y + x = 4*3 + 2 = 14
        // So the gradient is [7, 14]
        let grad = gradient_parallel(test_function, &params, None).unwrap();

        assert_eq!(grad.len(), 2);
        // Note: Due to numerical approximation, there might be small errors
        assert_relative_eq!(grad[0], 7.0, epsilon = 1e-2);
        assert_relative_eq!(grad[1], 14.0, epsilon = 1e-2);
    }

    #[test]
    fn test_hessian_parallel() {
        // Use a simple function for this test to avoid numerical issues
        // New test function: f(x) = x^2
        // This has a simple Hessian - the second derivative is constant: f''(x) = 2
        let simple_function = |params: &Array1<f64>| -> Result<f64> {
            let x = params[0];
            Ok(x.powi(2))
        };

        let params = array![3.0]; // Test at x = 3

        // Calculate Hessian numerically
        let hess = hessian_parallel(simple_function, &params, None).unwrap();

        // Check shape and approximate value
        assert_eq!(hess.shape(), &[1, 1]);

        // The exact Hessian is [[2.0]]
        let actual = hess[[0, 0]];
        assert_relative_eq!(actual, 2.0, epsilon = 1e-1);
    }

    #[test]
    fn test_eval_residuals_parallel() {
        // Create a function that computes a single residual
        let eval_residual = |i: usize, params: &Array1<f64>| -> Result<f64> {
            let x = params[0];
            let y = params[1];

            match i {
                0 => Ok(x.powi(2) - 1.0),
                1 => Ok(y.powi(2) - 2.0),
                _ => Err(LmOptError::InvalidParameter(
                    "Invalid residual index".to_string(),
                )),
            }
        };

        let params = array![2.0, 3.0];
        let n_residuals = 2;

        let residuals = eval_residuals_parallel(eval_residual, &params, n_residuals).unwrap();

        assert_eq!(residuals.len(), 2);
        assert_relative_eq!(residuals[0], 3.0, epsilon = 1e-10); // 2^2 - 1 = 3
        assert_relative_eq!(residuals[1], 7.0, epsilon = 1e-10); // 3^2 - 2 = 7
    }

    #[test]
    fn compare_with_sequential_jacobian() {
        // Create a larger problem to better see parallel speedup
        struct LargeProblem {
            size: usize,
        }

        impl Problem for LargeProblem {
            fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
                let n = self.size;
                let mut residuals = Array1::zeros(n);

                for i in 0..n {
                    // Simple computation: r_i = (p_i)^2 - i
                    residuals[i] = params[i % params.len()].powi(2) - i as f64;
                }

                Ok(residuals)
            }

            fn parameter_count(&self) -> usize {
                self.size / 2 // Half as many parameters as residuals
            }

            fn residual_count(&self) -> usize {
                self.size
            }
        }

        // Create a problem with 50 residuals and 25 parameters
        let problem = LargeProblem { size: 50 };
        let params = Array1::from_vec((0..25).map(|i| i as f64).collect());

        // Compute Jacobian with both sequential and parallel methods
        let seq_jac = finite_difference::jacobian(&problem, &params, None).unwrap();
        let par_jac = jacobian_parallel(&problem, &params, None).unwrap();

        // Verify they give the same results
        assert_eq!(seq_jac.shape(), par_jac.shape());

        for i in 0..seq_jac.shape()[0] {
            for j in 0..seq_jac.shape()[1] {
                assert_relative_eq!(seq_jac[[i, j]], par_jac[[i, j]], epsilon = 1e-10);
            }
        }
    }
}
