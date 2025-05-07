//! Trust region implementation for the Levenberg-Marquardt algorithm.
//!
//! This module provides a trust region implementation that adapts the step size
//! based on the agreement between predicted and actual reduction in cost.

use crate::error::Result;
use ndarray::Array1;

/// Trust region implementation for the Levenberg-Marquardt algorithm.
pub struct TrustRegion {
    /// Current value of the damping parameter
    pub lambda: f64,

    /// Minimum allowed value for the damping parameter
    pub lambda_min: f64,

    /// Maximum allowed value for the damping parameter
    pub lambda_max: f64,

    /// Factor to increase lambda by when step is rejected
    pub lambda_increase_factor: f64,

    /// Factor to decrease lambda by when step is accepted
    pub lambda_decrease_factor: f64,

    /// Minimum gain ratio required to accept a step
    pub min_gain_ratio: f64,

    /// Maximum gain ratio above which to decrease lambda
    pub good_gain_ratio: f64,
}

impl Default for TrustRegion {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            lambda_min: 1e-10,
            lambda_max: 1e10,
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            min_gain_ratio: 1e-3,
            good_gain_ratio: 0.75,
        }
    }
}

impl TrustRegion {
    /// Creates a new TrustRegion with default parameters.
    pub fn new() -> Self {
        Default::default()
    }

    /// Updates the damping parameter based on the gain ratio.
    ///
    /// # Arguments
    ///
    /// * `gain_ratio` - The ratio of actual reduction to predicted reduction
    pub fn update_lambda(&mut self, gain_ratio: f64) -> bool {
        if gain_ratio > self.min_gain_ratio {
            // Step accepted
            if gain_ratio > self.good_gain_ratio {
                // Good step, decrease lambda
                self.lambda = (self.lambda * self.lambda_decrease_factor).max(self.lambda_min);
            }
            true
        } else {
            // Step rejected, increase lambda
            self.lambda = (self.lambda * self.lambda_increase_factor).min(self.lambda_max);
            false
        }
    }

    /// Calculates the gain ratio between actual and predicted reduction.
    ///
    /// # Arguments
    ///
    /// * `current_cost` - The current cost function value
    /// * `new_cost` - The new cost function value after the step
    /// * `predicted_reduction` - The predicted reduction in cost
    ///
    /// # Returns
    ///
    /// * The gain ratio (actual reduction / predicted reduction)
    pub fn gain_ratio(current_cost: f64, new_cost: f64, predicted_reduction: f64) -> f64 {
        let actual_reduction = current_cost - new_cost;

        if predicted_reduction.abs() < 1e-10 {
            if actual_reduction.abs() < 1e-10 {
                1.0
            } else {
                0.0
            }
        } else {
            actual_reduction / predicted_reduction
        }
    }

    /// Resets the trust region parameters to their default values.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
