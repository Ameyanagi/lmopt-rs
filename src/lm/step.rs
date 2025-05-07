//! Step calculation for the Levenberg-Marquardt algorithm.
//!
//! This module provides functionality for computing the Levenberg-Marquardt step,
//! which combines the Gauss-Newton and gradient descent steps.

use crate::error::{LmOptError, Result};
use crate::lm::trust_region::TrustRegion;
use ndarray::{Array1, Array2};

/// Result of a Levenberg-Marquardt step calculation.
pub struct StepResult {
    /// The calculated step vector
    pub step: Array1<f64>,

    /// The predicted reduction in cost function value
    pub predicted_reduction: f64,

    /// The damping parameter used to calculate the step
    pub lambda: f64,
}

/// Handles step calculation for the Levenberg-Marquardt algorithm.
pub struct LmStep;

impl LmStep {
    /// Calculates the Levenberg-Marquardt step using a specified decomposition method.
    ///
    /// # Arguments
    ///
    /// * `jacobian` - The Jacobian matrix at the current position
    /// * `residuals` - The residuals at the current position
    /// * `trust_region` - The trust region controller
    /// * `decomposition_method` - The method to use for matrix decomposition
    ///
    /// # Returns
    ///
    /// * The step result containing the step vector and other information
    pub fn calculate_step(
        jacobian: &Array2<f64>,
        residuals: &Array1<f64>,
        trust_region: &TrustRegion,
    ) -> Result<StepResult> {
        // Compute J^T * J
        let j_t_j = jacobian.t().dot(jacobian);

        // Compute J^T * r
        let j_t_r = jacobian.t().dot(residuals);

        // Add lambda to the diagonal of J^T * J
        let mut augmented_j_t_j = j_t_j.clone();
        for i in 0..augmented_j_t_j.nrows() {
            augmented_j_t_j[[i, i]] += trust_region.lambda * augmented_j_t_j[[i, i]].max(1e-10);
        }

        // Solve the system (J^T * J + lambda * D) * step = -J^T * r
        // Where D is a diagonal matrix with the same diagonal as J^T * J
        let step = match LmStep::solve_cholesky(&augmented_j_t_j, &-&j_t_r) {
            Ok(step) => step,
            Err(_) => {
                // If Cholesky decomposition fails, fall back to a simpler method
                // This is a simple gradient descent step as a fallback
                -&j_t_r * (1.0 / (trust_region.lambda + 1.0))
            }
        };

        // Calculate the predicted reduction in cost
        let predicted_reduction =
            LmStep::predicted_reduction(&j_t_j, &j_t_r, &step, trust_region.lambda);

        Ok(StepResult {
            step,
            predicted_reduction,
            lambda: trust_region.lambda,
        })
    }

    /// Solves the linear system A * x = b using Cholesky decomposition.
    ///
    /// # Arguments
    ///
    /// * `a` - The coefficient matrix
    /// * `b` - The right-hand side vector
    ///
    /// # Returns
    ///
    /// * The solution vector
    fn solve_cholesky(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        use crate::utils::matrix_convert::{faer_vec_to_ndarray, ndarray_to_faer};
        use faer::Mat;

        // Convert ndarray matrix to faer matrix
        let a_faer = ndarray_to_faer(a);

        // Convert right-hand side vector to faer vector
        let mut b_faer = Mat::zeros(b.len(), 1);
        for i in 0..b.len() {
            b_faer[(i, 0)] = b[i];
        }

        // Use a simpler method: solve the system using direct solve
        // This is more stable for our purposes
        let maybe_x = a_faer.solve_into_to_req(b_faer);

        let x = match maybe_x {
            Some(x) => x,
            None => {
                return Err(LmOptError::LinearAlgebraError(
                    "Linear system solution failed".to_string(),
                ));
            }
        };

        // Convert solution back to ndarray
        Ok(faer_vec_to_ndarray(&x))
    }

    /// Calculates the predicted reduction in cost function value for a step.
    ///
    /// # Arguments
    ///
    /// * `j_t_j` - The J^T * J matrix
    /// * `j_t_r` - The J^T * r vector
    /// * `step` - The calculated step vector
    /// * `lambda` - The damping parameter
    ///
    /// # Returns
    ///
    /// * The predicted reduction in cost function value
    fn predicted_reduction(
        j_t_j: &Array2<f64>,
        j_t_r: &Array1<f64>,
        step: &Array1<f64>,
        lambda: f64,
    ) -> f64 {
        let model_reduction = -step.dot(j_t_r);
        let step_size_penalty = 0.5 * lambda * step.dot(&(step * step));
        model_reduction - step_size_penalty
    }
}
