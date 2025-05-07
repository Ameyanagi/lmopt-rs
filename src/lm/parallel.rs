//! Parallel implementations of the Levenberg-Marquardt algorithm.
//!
//! This module provides parallel versions of the Levenberg-Marquardt algorithm
//! for problems where parallel processing can significantly improve performance.

use faer::{Col, Mat};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

use crate::error::{LmOptError, Result};
use crate::problem::Problem;
use crate::utils::matrix_convert::{
    faer_to_ndarray, faer_vec_to_ndarray, ndarray_to_faer, ndarray_vec_to_faer,
};
use crate::utils::parallel::{jacobian_central_parallel, jacobian_parallel};

use super::algorithm::LmResult;
use super::config::{DecompositionMethod, DiffMethod, LmConfig};
use super::robust::RobustLoss;

/// The parallel Levenberg-Marquardt optimizer.
///
/// This implementation uses Rayon for parallel processing of the most computationally
/// intensive parts of the algorithm, such as Jacobian calculation and matrix operations.
#[derive(Debug, Clone)]
pub struct ParallelLevenbergMarquardt {
    /// Configuration options
    config: LmConfig,
}

impl ParallelLevenbergMarquardt {
    /// Create a new parallel Levenberg-Marquardt optimizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: LmConfig::default(),
        }
    }

    /// Create a new parallel Levenberg-Marquardt optimizer with the given configuration.
    pub fn with_config(config: LmConfig) -> Self {
        Self { config }
    }

    /// Create a new parallel Levenberg-Marquardt optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self {
            config: LmConfig::default(),
        }
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.config.max_iterations = max_iterations;
        self
    }

    /// Set the tolerance for change in residual norm.
    pub fn with_ftol(mut self, ftol: f64) -> Self {
        self.config.ftol = ftol;
        self
    }

    /// Set the tolerance for change in parameter values.
    pub fn with_xtol(mut self, xtol: f64) -> Self {
        self.config.xtol = xtol;
        self
    }

    /// Set the tolerance for gradient norm.
    pub fn with_gtol(mut self, gtol: f64) -> Self {
        self.config.gtol = gtol;
        self
    }

    /// Set the initial value for the damping parameter.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.config.initial_lambda = lambda;
        self
    }

    /// Set the factor by which to increase lambda.
    pub fn with_lambda_up_factor(mut self, factor: f64) -> Self {
        self.config.lambda_up_factor = factor;
        self
    }

    /// Set the factor by which to decrease lambda.
    pub fn with_lambda_down_factor(mut self, factor: f64) -> Self {
        self.config.lambda_down_factor = factor;
        self
    }

    /// Set the minimum value for lambda.
    pub fn with_min_lambda(mut self, min_lambda: f64) -> Self {
        self.config.min_lambda = min_lambda;
        self
    }

    /// Set the maximum value for lambda.
    pub fn with_max_lambda(mut self, max_lambda: f64) -> Self {
        self.config.max_lambda = max_lambda;
        self
    }

    /// Set the method used for calculating the Jacobian.
    pub fn with_differentiation_method(mut self, method: DiffMethod) -> Self {
        self.config.diff_method = method;
        self
    }

    /// Set the method used for solving the linear system.
    pub fn with_decomposition_method(mut self, method: DecompositionMethod) -> Self {
        self.config.decomposition_method = method;
        self
    }

    /// Set whether to calculate and return the Jacobian at the solution.
    pub fn with_calc_jacobian(mut self, calc_jacobian: bool) -> Self {
        self.config.calc_jacobian = calc_jacobian;
        self
    }

    /// Set the robust loss function to use for outlier-resistant fitting.
    pub fn with_loss_function(mut self, loss_function: RobustLoss) -> Self {
        self.config.loss_function = loss_function;
        self
    }

    /// Set the number of iterations of iteratively reweighted least squares (IRLS) for robust fitting.
    pub fn with_robust_iterations(mut self, iterations: usize) -> Self {
        self.config.robust_iterations = iterations;
        self
    }

    /// Minimize the sum of squared residuals for the given problem using parallel processing.
    ///
    /// This method implements the Levenberg-Marquardt algorithm with parallel processing
    /// for the most computationally intensive parts.
    ///
    /// # Arguments
    ///
    /// * `problem` - The problem to solve
    /// * `initial_params` - Initial guess for the parameter values
    ///
    /// # Returns
    ///
    /// * `Result<LmResult>` - The result of the optimization
    pub fn minimize<P: Problem + Sync>(
        &self,
        problem: &P,
        initial_params: Array1<f64>,
    ) -> Result<LmResult> {
        // Check parameter dimensions
        let n_params = problem.parameter_count();
        if initial_params.len() != n_params {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                n_params,
                initial_params.len()
            )));
        }

        // Initialize parameters and damping
        let mut params = initial_params;
        let mut lambda = self.config.initial_lambda;

        // Evaluate initial residuals and cost
        let mut residuals = problem.eval(&params)?;
        let mut func_evals = 1;

        // Initialize robust weights if needed
        let mut robust_weights = if matches!(self.config.loss_function, RobustLoss::LeastSquares) {
            None
        } else {
            Some(Array1::ones(residuals.len()))
        };

        // Apply robust loss if specified
        let (weighted_residuals, cost) = if let Some(weights) = &robust_weights {
            // Initial robust fit should use least squares before we have good residuals
            // So we start with all weights = 1
            let weighted = &residuals * weights.mapv(f64::sqrt);
            let cost = weighted.iter().map(|r| r.powi(2)).sum();
            (weighted, cost)
        } else {
            let cost = residuals.iter().map(|r| r.powi(2)).sum();
            (residuals.clone(), cost)
        };

        // Iterate until convergence or max iterations
        let mut iterations = 0;
        let mut jacobian: Option<Array2<f64>> = None;

        // For robust fitting, do multiple rounds of IRLS (Iteratively Reweighted Least Squares)
        for robust_iter in 0..self.config.robust_iterations.max(1) {
            // If we have a robust loss function, update weights
            if robust_iter > 0 && !matches!(self.config.loss_function, RobustLoss::LeastSquares) {
                let (_, weights) = self.config.loss_function.apply(&residuals);
                robust_weights = Some(weights);
                // Recalculate weighted residuals
                let weighted = &residuals * robust_weights.as_ref().unwrap().mapv(f64::sqrt);
                let new_cost = weighted.iter().map(|r| r.powi(2)).sum();
                // Use the weighted residuals for the rest of this IRLS iteration
                residuals = weighted;
                lambda = self.config.initial_lambda; // Reset lambda for the new weighted problem
            }

            // Main LM iteration loop
            loop {
                // Compute Jacobian using parallelized methods
                jacobian = Some(match self.config.diff_method {
                    DiffMethod::Forward => jacobian_parallel(problem, &params, None)?,
                    DiffMethod::Central => jacobian_central_parallel(problem, &params, None)?,
                    DiffMethod::Auto => {
                        if problem.has_custom_jacobian() {
                            problem.jacobian(&params)?
                        } else {
                            jacobian_parallel(problem, &params, None)?
                        }
                    }
                });

                func_evals += match self.config.diff_method {
                    DiffMethod::Forward => n_params,
                    DiffMethod::Central => 2 * n_params,
                    DiffMethod::Auto => {
                        if problem.has_custom_jacobian() {
                            1 // Custom Jacobian typically counts as one evaluation
                        } else {
                            n_params
                        }
                    }
                };

                let j = ndarray_to_faer(jacobian.as_ref().unwrap())?;

                // Apply robust weights to Jacobian if needed
                let j_weighted = if let Some(weights) = &robust_weights {
                    // Create a diagonal matrix of square root of weights
                    let n = weights.len();
                    let mut w_mat = Mat::zeros(n, n);

                    // Parallelize the creation of the weight matrix diagonal
                    (0..n).into_par_iter().for_each(|i| {
                        w_mat[(i, i)] = weights[i].sqrt();
                    });

                    // Apply weights to Jacobian: W^(1/2) * J
                    w_mat * &j
                } else {
                    j
                };

                let r = ndarray_vec_to_faer(&residuals)?;

                // Compute gradient g = J^T * r
                let g = j_weighted.transpose() * &r;

                // Check gradient convergence
                let gradient_norm = g.norm_l2();
                if gradient_norm < self.config.gtol {
                    return Ok(LmResult {
                        params: params.clone(),
                        residuals: if let Some(weights) = &robust_weights {
                            problem.eval(&params)?
                        } else {
                            residuals
                        },
                        cost,
                        iterations,
                        func_evals,
                        success: true,
                        message: format!(
                            "Gradient convergence: ||g|| = {:.2e} < {:.2e}",
                            gradient_norm, self.config.gtol
                        ),
                        jacobian: if self.config.calc_jacobian {
                            Some(match self.config.diff_method {
                                DiffMethod::Forward => jacobian_parallel(problem, &params, None)?,
                                DiffMethod::Central => {
                                    jacobian_central_parallel(problem, &params, None)?
                                }
                                DiffMethod::Auto => {
                                    if problem.has_custom_jacobian() {
                                        problem.jacobian(&params)?
                                    } else {
                                        jacobian_parallel(problem, &params, None)?
                                    }
                                }
                            })
                        } else {
                            None
                        },
                        robust_weights: robust_weights.clone(),
                    });
                }

                // Calculate step using trust region approach
                let step = match self.calculate_step(&j_weighted, &r, lambda)? {
                    Some(s) => s,
                    None => {
                        // If step calculation failed, increase lambda and try again
                        lambda =
                            (lambda * self.config.lambda_up_factor).min(self.config.max_lambda);
                        if lambda == self.config.max_lambda {
                            return Err(LmOptError::ConvergenceFailure(
                                "Failed to calculate step, and lambda reached maximum".to_string(),
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
                func_evals += 1;

                // Apply robust weighting if needed
                let (new_weighted_residuals, new_cost) = if let Some(weights) = &robust_weights {
                    let weighted = &new_residuals * weights.mapv(f64::sqrt);
                    let cost = weighted.iter().map(|r| r.powi(2)).sum();
                    (weighted, cost)
                } else {
                    let cost = new_residuals.iter().map(|r| r.powi(2)).sum();
                    (new_residuals.clone(), cost)
                };

                // Check if the step reduced the cost
                if new_cost < cost {
                    // Step accepted

                    // Check convergence using parallel array operations
                    let param_diffs: Vec<_> = params
                        .iter()
                        .zip(new_params.iter())
                        .map(|(&p, &np)| (np - p).abs())
                        .collect();

                    let param_change = param_diffs
                        .into_par_iter()
                        .fold(
                            || 0.0 / 0.0,    // Initial fold value (NaN)
                            |a, b| a.max(b), // Max function
                        )
                        .reduce(
                            || 0.0 / 0.0,    // Initial reduce value (NaN)
                            |a, b| a.max(b), // Max function
                        );

                    let cost_change = (cost - new_cost) / cost.max(1e-10);

                    let status = if iterations >= self.config.max_iterations {
                        "Maximum iterations ({}) reached".to_string()
                    } else if param_change < self.config.xtol {
                        return Ok(LmResult {
                            params: new_params.clone(),
                            residuals: problem.eval(&new_params)?,
                            cost: new_cost,
                            iterations: iterations + 1,
                            func_evals,
                            success: true,
                            message: format!(
                                "Parameter convergence: |dx|/|x| = {:.2e} < {:.2e}",
                                param_change, self.config.xtol
                            ),
                            jacobian: if self.config.calc_jacobian {
                                Some(match self.config.diff_method {
                                    DiffMethod::Forward => {
                                        jacobian_parallel(problem, &new_params, None)?
                                    }
                                    DiffMethod::Central => {
                                        jacobian_central_parallel(problem, &new_params, None)?
                                    }
                                    DiffMethod::Auto => {
                                        if problem.has_custom_jacobian() {
                                            problem.jacobian(&new_params)?
                                        } else {
                                            jacobian_parallel(problem, &new_params, None)?
                                        }
                                    }
                                })
                            } else {
                                None
                            },
                            robust_weights: robust_weights.clone(),
                        });
                    } else if cost_change < self.config.ftol {
                        return Ok(LmResult {
                            params: new_params.clone(),
                            residuals: problem.eval(&new_params)?,
                            cost: new_cost,
                            iterations: iterations + 1,
                            func_evals,
                            success: true,
                            message: format!(
                                "Cost convergence: |df|/|f| = {:.2e} < {:.2e}",
                                cost_change, self.config.ftol
                            ),
                            jacobian: if self.config.calc_jacobian {
                                Some(match self.config.diff_method {
                                    DiffMethod::Forward => {
                                        jacobian_parallel(problem, &new_params, None)?
                                    }
                                    DiffMethod::Central => {
                                        jacobian_central_parallel(problem, &new_params, None)?
                                    }
                                    DiffMethod::Auto => {
                                        if problem.has_custom_jacobian() {
                                            problem.jacobian(&new_params)?
                                        } else {
                                            jacobian_parallel(problem, &new_params, None)?
                                        }
                                    }
                                })
                            } else {
                                None
                            },
                            robust_weights: robust_weights.clone(),
                        });
                    } else {
                        "".to_string()
                    };

                    // Update state
                    params = new_params;
                    residuals = new_weighted_residuals;
                    lambda = (lambda * self.config.lambda_down_factor).max(self.config.min_lambda);
                    iterations += 1;

                    // Handle max iterations reached
                    if !status.is_empty() {
                        return Ok(LmResult {
                            params,
                            residuals: problem.eval(&params)?,
                            cost: new_cost,
                            iterations,
                            func_evals,
                            success: false,
                            message: status,
                            jacobian: if self.config.calc_jacobian {
                                Some(match self.config.diff_method {
                                    DiffMethod::Forward => {
                                        jacobian_parallel(problem, &params, None)?
                                    }
                                    DiffMethod::Central => {
                                        jacobian_central_parallel(problem, &params, None)?
                                    }
                                    DiffMethod::Auto => {
                                        if problem.has_custom_jacobian() {
                                            problem.jacobian(&params)?
                                        } else {
                                            jacobian_parallel(problem, &params, None)?
                                        }
                                    }
                                })
                            } else {
                                None
                            },
                            robust_weights: robust_weights.clone(),
                        });
                    }
                } else {
                    // Step rejected - increase lambda and try again
                    lambda = (lambda * self.config.lambda_up_factor).min(self.config.max_lambda);

                    if lambda == self.config.max_lambda {
                        return Ok(LmResult {
                            params,
                            residuals: problem.eval(&params)?,
                            cost,
                            iterations,
                            func_evals,
                            success: false,
                            message: "Failed to decrease cost, and lambda reached maximum"
                                .to_string(),
                            jacobian: if self.config.calc_jacobian {
                                Some(match self.config.diff_method {
                                    DiffMethod::Forward => {
                                        jacobian_parallel(problem, &params, None)?
                                    }
                                    DiffMethod::Central => {
                                        jacobian_central_parallel(problem, &params, None)?
                                    }
                                    DiffMethod::Auto => {
                                        if problem.has_custom_jacobian() {
                                            problem.jacobian(&params)?
                                        } else {
                                            jacobian_parallel(problem, &params, None)?
                                        }
                                    }
                                })
                            } else {
                                None
                            },
                            robust_weights: robust_weights.clone(),
                        });
                    }
                }
            }

            // If we've completed an IRLS iteration, recalculate the unweighted residuals
            residuals = problem.eval(&params)?;
            func_evals += 1;
        }

        // If we get here, the optimization converged
        Ok(LmResult {
            params,
            residuals: problem.eval(&params)?,
            cost,
            iterations,
            func_evals,
            success: true,
            message: "Optimization converged successfully".to_string(),
            jacobian: if self.config.calc_jacobian {
                Some(match self.config.diff_method {
                    DiffMethod::Forward => jacobian_parallel(problem, &params, None)?,
                    DiffMethod::Central => jacobian_central_parallel(problem, &params, None)?,
                    DiffMethod::Auto => {
                        if problem.has_custom_jacobian() {
                            problem.jacobian(&params)?
                        } else {
                            jacobian_parallel(problem, &params, None)?
                        }
                    }
                })
            } else {
                None
            },
            robust_weights,
        })
    }

    /// Calculate the Levenberg-Marquardt step using parallel decomposition methods when possible.
    ///
    /// This method solves the equation (J^T J + λI) δ = J^T r, using parallel
    /// matrix operations where available.
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

        // Add damping to diagonal (need to modify each value individually due to borrowing rules)
        for i in 0..n {
            a_ndarray[[i, i]] += lambda;
        }

        // Compute J^T r
        let jtr_ndarray = jt_ndarray.dot(&r_ndarray);

        // Solve the system based on the chosen decomposition method
        let step_ndarray = match self.config.decomposition_method {
            // We'll only implement parallel versions of QR and SVD since Cholesky
            // implementation is more sequential by nature
            DecompositionMethod::Cholesky => self.solve_cholesky(&a_ndarray, &jtr_ndarray)?,
            DecompositionMethod::QR => self.solve_qr_parallel(&j_ndarray, &jtr_ndarray)?,
            DecompositionMethod::SVD => self.solve_svd_parallel(&j_ndarray, &jtr_ndarray)?,
            DecompositionMethod::Auto => {
                // Try Cholesky first, fall back to QR if it fails
                match self.solve_cholesky(&a_ndarray, &jtr_ndarray) {
                    Ok(step) => step,
                    Err(_) => self.solve_qr_parallel(&j_ndarray, &jtr_ndarray)?,
                }
            }
        };

        // Convert back to faer
        let step = ndarray_vec_to_faer(&(-step_ndarray))?;
        Ok(Some(step))
    }

    /// Solve the system using Cholesky decomposition.
    fn solve_cholesky(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.shape()[0];

        // Simple Cholesky decomposition (without pivoting)
        let mut a_mutable = a.clone();
        let mut cholesky_success = true;

        for k in 0..n {
            // Diagonal element
            for j in 0..k {
                a_mutable[[k, k]] -= a_mutable[[k, j]] * a_mutable[[k, j]];
            }

            if a_mutable[[k, k]] <= 0.0 {
                return Err(LmOptError::LinearSolverFailure(
                    "Cholesky decomposition failed - matrix not positive definite".to_string(),
                ));
            }

            let akk_sqrt = a_mutable[[k, k]].sqrt();
            a_mutable[[k, k]] = akk_sqrt;

            // Update column elements
            for i in k + 1..n {
                for j in 0..k {
                    a_mutable[[i, k]] -= a_mutable[[i, j]] * a_mutable[[k, j]];
                }
                a_mutable[[i, k]] /= akk_sqrt;
            }
        }

        // Cholesky succeeded, so use it to solve the system
        let mut rhs = b.clone();
        let mut solution = Array1::zeros(n);

        // Forward substitution (L * y = b)
        for i in 0..n {
            for j in 0..i {
                rhs[i] -= a_mutable[[i, j]] * rhs[j];
            }
            rhs[i] /= a_mutable[[i, i]];
        }

        // Backward substitution (L^T * x = y)
        for i in (0..n).rev() {
            solution[i] = rhs[i];
            for j in (i + 1)..n {
                solution[i] -= a_mutable[[j, i]] * solution[j];
            }
            solution[i] /= a_mutable[[i, i]];
        }

        Ok(solution)
    }

    /// Solve the system using QR decomposition with parallel processing.
    fn solve_qr_parallel(&self, j: &Array2<f64>, jtr: &Array1<f64>) -> Result<Array1<f64>> {
        let m = j.shape()[0]; // Number of rows (observations)
        let n = j.shape()[1]; // Number of columns (parameters)

        // Initialize Q and R matrices
        let mut q = Array2::zeros((m, n));
        let mut r = Array2::zeros((n, n));

        // QR decomposition with modified Gram-Schmidt process
        for j_idx in 0..n {
            // Copy column j of A to Q
            let j_col = j.column(j_idx).to_owned();
            for i in 0..m {
                q[[i, j_idx]] = j_col[i];
            }

            // Orthogonalize column j with respect to previous columns
            // This step processes each previous column sequentially
            for k in 0..j_idx {
                // Compute dot product in parallel for large vectors
                let dot_product = if m > 1000 {
                    // For large vectors, use parallel reduction
                    let chunks: Vec<_> = (0..m).collect();
                    chunks.par_iter().map(|&i| q[[i, k]] * q[[i, j_idx]]).sum()
                } else {
                    // For small vectors, use sequential sum
                    (0..m).map(|i| q[[i, k]] * q[[i, j_idx]]).sum()
                };

                r[[k, j_idx]] = dot_product;

                // Update column j in parallel for large vectors
                if m > 1000 {
                    (0..m).into_par_iter().for_each(|i| {
                        q[[i, j_idx]] -= r[[k, j_idx]] * q[[i, k]];
                    });
                } else {
                    for i in 0..m {
                        q[[i, j_idx]] -= r[[k, j_idx]] * q[[i, k]];
                    }
                }
            }

            // Normalize column j
            // Compute norm in parallel for large vectors
            let norm_squared = if m > 1000 {
                (0..m)
                    .into_par_iter()
                    .map(|i| q[[i, j_idx]] * q[[i, j_idx]])
                    .sum()
            } else {
                (0..m).map(|i| q[[i, j_idx]] * q[[i, j_idx]]).sum()
            };

            let norm = norm_squared.sqrt();

            if norm > 1e-10 {
                r[[j_idx, j_idx]] = norm;

                // Normalize column in parallel for large vectors
                if m > 1000 {
                    (0..m).into_par_iter().for_each(|i| {
                        q[[i, j_idx]] /= norm;
                    });
                } else {
                    for i in 0..m {
                        q[[i, j_idx]] /= norm;
                    }
                }
            } else {
                // Column is linearly dependent - set it to zero
                if m > 1000 {
                    (0..m).into_par_iter().for_each(|i| {
                        q[[i, j_idx]] = 0.0;
                    });
                } else {
                    for i in 0..m {
                        q[[i, j_idx]] = 0.0;
                    }
                }
                r[[j_idx, j_idx]] = 0.0;
            }
        }

        // Compute Q^T * b
        let mut qty = Array1::zeros(n);

        // Compute dot products in parallel for each row of Q
        let qty_values: Vec<(usize, f64)> = (0..n)
            .into_par_iter()
            .map(|j| {
                let dot = if m > 1000 {
                    (0..m).into_par_iter().map(|i| q[[i, j]] * jtr[i]).sum()
                } else {
                    (0..m).map(|i| q[[i, j]] * jtr[i]).sum()
                };
                (j, dot)
            })
            .collect();

        // Fill in the qty vector
        for (j, value) in qty_values {
            qty[j] = value;
        }

        // Solve R * x = Q^T * b using back-substitution
        let mut x = Array1::zeros(n);
        for j in (0..n).rev() {
            if r[[j, j]].abs() < 1e-10 {
                // Handle rank deficiency - use a small value for diagonal
                x[j] = 0.0;
            } else {
                x[j] = qty[j];
                for k in (j + 1)..n {
                    x[j] -= r[[j, k]] * x[k];
                }
                x[j] /= r[[j, j]];
            }
        }

        Ok(x)
    }

    /// Solve the system using SVD decomposition with parallel processing.
    fn solve_svd_parallel(&self, j: &Array2<f64>, jtr: &Array1<f64>) -> Result<Array1<f64>> {
        // For now, we'll use QR as a fallback
        // In a future implementation, we could use a proper SVD implementation
        // with parallel processing
        self.solve_qr_parallel(j, jtr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::robust::RobustLoss;
    use super::*;
    use crate::lm::algorithm::LevenbergMarquardt;
    use crate::problem::Problem;
    use approx::assert_relative_eq;
    use ndarray::{array, Array1, Array2};

    /// A simple linear model for testing: f(x) = a * x + b
    #[derive(Clone)]
    struct LinearModel {
        x_data: Array1<f64>,
        y_data: Array1<f64>,
        eval_count: std::sync::atomic::AtomicUsize,
        jacobian_count: std::sync::atomic::AtomicUsize,
    }

    impl LinearModel {
        fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
            assert_eq!(
                x_data.len(),
                y_data.len(),
                "x and y data must have the same length"
            );
            Self {
                x_data,
                y_data,
                eval_count: std::sync::atomic::AtomicUsize::new(0),
                jacobian_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn get_eval_count(&self) -> usize {
            self.eval_count.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn get_jacobian_count(&self) -> usize {
            self.jacobian_count
                .load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl Problem for LinearModel {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            self.eval_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if params.len() != 2 {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected 2 parameters, got {}",
                    params.len()
                )));
            }

            let a = params[0];
            let b = params[1];

            let residuals = self
                .x_data
                .iter()
                .zip(self.y_data.iter())
                .map(|(x, y)| a * x + b - y)
                .collect::<Vec<f64>>();

            Ok(Array1::from_vec(residuals))
        }

        fn parameter_count(&self) -> usize {
            2 // a and b
        }

        fn residual_count(&self) -> usize {
            self.x_data.len()
        }

        // Custom Jacobian implementation for the linear model
        fn jacobian(&self, _params: &Array1<f64>) -> Result<Array2<f64>> {
            self.jacobian_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

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
    fn test_parallel_linear_fit() {
        // Create test data: y = 2x + 3 + noise
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.1, 7.0, 8.9, 11.2, 13.0]; // Approximately 2x + 3

        let model = LinearModel::new(x, y);
        let initial_params = array![1.0, 1.0]; // Initial guess [a, b] = [1, 1]

        let lm = ParallelLevenbergMarquardt::with_default_config();
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
    fn compare_parallel_vs_sequential() {
        // Create a larger dataset to better demonstrate parallel performance
        let n_points = 1000;
        let mut x = Array1::zeros(n_points);
        let mut y = Array1::zeros(n_points);

        for i in 0..n_points {
            let x_val = i as f64 / 100.0;
            x[i] = x_val;
            y[i] = 2.5 * x_val + 1.7 + 0.1 * (rand::random::<f64>() - 0.5);
        }

        let model_seq = LinearModel::new(x.clone(), y.clone());
        let model_par = model_seq.clone();
        let initial_params = array![1.0, 1.0];

        // Run sequential optimizer
        let lm_seq = LevenbergMarquardt::with_default_config();
        let result_seq = lm_seq.minimize(&model_seq, initial_params.clone()).unwrap();

        // Run parallel optimizer
        let lm_par = ParallelLevenbergMarquardt::with_default_config();
        let result_par = lm_par.minimize(&model_par, initial_params).unwrap();

        // Check that both methods converged to the same solution
        assert_relative_eq!(result_seq.params[0], result_par.params[0], epsilon = 1e-2);
        assert_relative_eq!(result_seq.params[1], result_par.params[1], epsilon = 1e-2);

        // Both should be close to the true values [2.5, 1.7]
        assert_relative_eq!(result_par.params[0], 2.5, epsilon = 0.1);
        assert_relative_eq!(result_par.params[1], 1.7, epsilon = 0.1);

        // Function counts should be comparable
        println!("Sequential eval count: {}", model_seq.get_eval_count());
        println!("Parallel eval count: {}", model_par.get_eval_count());
    }

    #[test]
    fn test_parallel_robust_linear_fit() {
        // Create test data: y = 2x + 3 + noise, with an outlier
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = array![5.1, 7.0, 8.9, 11.2, 13.0]; // Approximately 2x + 3

        // Add an outlier
        y[2] = 20.0; // This should be around 9, so it's a large outlier

        let model = LinearModel::new(x.clone(), y.clone());
        let initial_params = array![1.0, 1.0]; // Initial guess [a, b] = [1, 1]

        // First try standard least squares
        let lm_std = ParallelLevenbergMarquardt::with_default_config();
        let result_std = lm_std.minimize(&model, initial_params.clone()).unwrap();

        // Now try with Huber loss
        let lm_robust = ParallelLevenbergMarquardt::new()
            .with_loss_function(RobustLoss::Huber(1.0))
            .with_robust_iterations(3);
        let result_robust = lm_robust.minimize(&model, initial_params).unwrap();

        // The robust fit should be closer to the true parameters [2, 3]
        println!(
            "Standard LS parameters: a={}, b={}",
            result_std.params[0], result_std.params[1]
        );
        println!(
            "Robust parameters: a={}, b={}",
            result_robust.params[0], result_robust.params[1]
        );

        // The robust fit should have parameters closer to [2, 3]
        let std_error = (result_std.params[0] - 2.0).abs() + (result_std.params[1] - 3.0).abs();
        let robust_error =
            (result_robust.params[0] - 2.0).abs() + (result_robust.params[1] - 3.0).abs();

        assert!(
            robust_error < std_error,
            "Robust fit error ({}) should be less than standard fit error ({})",
            robust_error,
            std_error
        );
    }
}
