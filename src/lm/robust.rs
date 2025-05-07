//! Robust fitting methods for handling outliers.
//!
//! This module provides robust loss functions and weighting schemes that can be
//! used with the Levenberg-Marquardt algorithm to reduce the influence of outliers.
//! Robust methods are essential for real-world data that may contain measurement errors,
//! fliers, or other anomalies that would otherwise disproportionately influence the fit.

use ndarray::Array1;
use std::fmt;

/// Different robust loss functions that can be used for outlier-resistant fitting.
#[derive(Debug, Clone, Copy)]
pub enum RobustLoss {
    /// Standard least squares (not robust, included as baseline)
    ///
    /// ρ(z) = z²
    LeastSquares,

    /// Huber loss function: quadratic for small residuals, linear for large ones
    ///
    /// ρ(z) = z² for |z| ≤ δ, 2δ|z| - δ² for |z| > δ
    Huber(f64),

    /// Soft L1 loss function: smooth approximation to absolute value
    ///
    /// ρ(z) = 2 * (sqrt(1 + z²) - 1)
    SoftL1,

    /// Cauchy or Lorentzian loss function: log-based function with tunable scale
    ///
    /// ρ(z) = log(1 + (z/c)²)
    Cauchy(f64),

    /// Arctan loss function: gradually decreasing influence
    ///
    /// ρ(z) = arctan(z²)
    Arctan,

    /// Tukey's biweight function: completely ignores large residuals
    ///
    /// ρ(z) = (1 - (1 - (z/c)²)³) for |z| ≤ c, 1 for |z| > c
    Tukey(f64),

    /// Custom loss function provided by the user
    Custom(Box<dyn Fn(f64) -> (f64, f64) + Send + Sync>),
}

impl Default for RobustLoss {
    fn default() -> Self {
        RobustLoss::LeastSquares
    }
}

impl fmt::Display for RobustLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RobustLoss::LeastSquares => write!(f, "Least Squares"),
            RobustLoss::Huber(delta) => write!(f, "Huber(delta={})", delta),
            RobustLoss::SoftL1 => write!(f, "Soft L1"),
            RobustLoss::Cauchy(c) => write!(f, "Cauchy(c={})", c),
            RobustLoss::Arctan => write!(f, "Arctan"),
            RobustLoss::Tukey(c) => write!(f, "Tukey(c={})", c),
            RobustLoss::Custom(_) => write!(f, "Custom"),
        }
    }
}

impl RobustLoss {
    /// Construct a new Huber loss function with the given delta parameter.
    ///
    /// The parameter delta controls the transition between quadratic and linear loss.
    ///
    /// # Arguments
    ///
    /// * `delta` - The threshold for switching between quadratic and linear loss
    pub fn huber(delta: f64) -> Self {
        RobustLoss::Huber(delta)
    }

    /// Construct a new Cauchy loss function with the given scale parameter.
    ///
    /// The parameter c controls the scale of the loss function.
    ///
    /// # Arguments
    ///
    /// * `c` - The scale parameter
    pub fn cauchy(c: f64) -> Self {
        RobustLoss::Cauchy(c)
    }

    /// Construct a new Tukey biweight loss function with the given c parameter.
    ///
    /// The parameter c controls where influence drops to zero.
    ///
    /// # Arguments
    ///
    /// * `c` - The parameter that determines where influence becomes zero
    pub fn tukey(c: f64) -> Self {
        RobustLoss::Tukey(c)
    }

    /// Construct a custom loss function.
    ///
    /// # Arguments
    ///
    /// * `func` - A function that takes a residual value and returns a tuple of
    ///   (rho(z), weight(z)) where rho is the loss function and weight is the
    ///   corresponding weight function
    pub fn custom<F>(func: F) -> Self
    where
        F: Fn(f64) -> (f64, f64) + Send + Sync + 'static,
    {
        RobustLoss::Custom(Box::new(func))
    }

    /// Evaluate the loss function and its corresponding weight for a given residual.
    ///
    /// # Arguments
    ///
    /// * `residual` - The residual value
    ///
    /// # Returns
    ///
    /// A tuple containing (rho(z), weight(z)) where:
    /// - rho(z) is the value of the loss function
    /// - weight(z) is the corresponding weight function value
    pub fn evaluate(&self, residual: f64) -> (f64, f64) {
        match *self {
            RobustLoss::LeastSquares => {
                let rho = residual * residual;
                let weight = 1.0;
                (rho, weight)
            }
            RobustLoss::Huber(delta) => {
                let abs_r = residual.abs();
                if abs_r <= delta {
                    // Quadratic region
                    let rho = residual * residual;
                    let weight = 1.0;
                    (rho, weight)
                } else {
                    // Linear region
                    let rho = 2.0 * delta * abs_r - delta * delta;
                    let weight = delta / abs_r;
                    (rho, weight)
                }
            }
            RobustLoss::SoftL1 => {
                let z_squared = residual * residual;
                let rho = 2.0 * ((1.0 + z_squared).sqrt() - 1.0);
                let weight = 1.0 / (1.0 + z_squared).sqrt();
                (rho, weight)
            }
            RobustLoss::Cauchy(c) => {
                let z_c = residual / c;
                let z_c_squared = z_c * z_c;
                let rho = (1.0 + z_c_squared).ln();
                let weight = 1.0 / (1.0 + z_c_squared);
                (rho, weight)
            }
            RobustLoss::Arctan => {
                let z_squared = residual * residual;
                let rho = z_squared.atan();
                let weight = 1.0 / (1.0 + z_squared);
                (rho, weight)
            }
            RobustLoss::Tukey(c) => {
                let z_c = residual / c;
                let abs_z_c = z_c.abs();

                if abs_z_c <= 1.0 {
                    let temp = 1.0 - z_c * z_c;
                    let temp_cubed = temp * temp * temp;
                    let rho = (1.0 - temp_cubed) * c * c / 6.0;
                    let weight = temp * temp;
                    (rho, weight)
                } else {
                    let rho = c * c / 6.0;
                    let weight = 0.0; // Zero weight for outliers
                    (rho, weight)
                }
            }
            RobustLoss::Custom(ref func) => func(residual),
        }
    }

    /// Apply the robust weighting to a vector of residuals.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The vector of residuals
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The sum of the robust loss function values
    /// - A vector of weights for each residual
    pub fn apply(&self, residuals: &Array1<f64>) -> (f64, Array1<f64>) {
        let mut loss_sum = 0.0;
        let mut weights = Array1::ones(residuals.len());

        for (i, &r) in residuals.iter().enumerate() {
            let (loss, weight) = self.evaluate(r);
            loss_sum += loss;
            weights[i] = weight;
        }

        (loss_sum, weights)
    }

    /// Compute weighted residuals by applying the robust loss function.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The original residuals
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The weighted residuals
    /// - The sum of the robust loss function values
    pub fn weighted_residuals(&self, residuals: &Array1<f64>) -> (Array1<f64>, f64) {
        let (loss_sum, weights) = self.apply(residuals);

        // Apply weights to residuals
        let weighted_residuals = residuals * weights.mapv(f64::sqrt);

        (weighted_residuals, loss_sum)
    }

    /// Estimate a good scale parameter for a given set of residuals.
    ///
    /// This is useful for automatically setting parameters like delta in Huber loss
    /// or c in Cauchy and Tukey losses.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The original residuals
    ///
    /// # Returns
    ///
    /// An appropriate scale parameter for the given residuals
    pub fn estimate_scale(residuals: &Array1<f64>) -> f64 {
        // Use median absolute deviation (MAD) as a robust estimate of scale
        let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
        abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Find the median
        let n = abs_residuals.len();
        let mad = if n % 2 == 0 {
            // Even number of elements - average the middle two
            let mid = n / 2;
            (abs_residuals[mid - 1] + abs_residuals[mid]) / 2.0
        } else {
            // Odd number of elements - take the middle one
            abs_residuals[n / 2]
        };

        // Scale MAD to be comparable to standard deviation for normal distribution
        // The factor 1.4826 makes MAD equivalent to standard deviation for normal distribution
        const MAD_TO_STD: f64 = 1.4826;
        mad * MAD_TO_STD
    }
}

/// A collection of recommended parameter values for robust loss functions based on
/// the estimated scale of the residuals.
impl RobustLoss {
    /// Create a Huber loss function with an automatically estimated parameter.
    ///
    /// The delta parameter is set to 1.345 times the estimated scale, which provides
    /// 95% efficiency for normally distributed errors.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The residuals used to estimate the scale
    pub fn auto_huber(residuals: &Array1<f64>) -> Self {
        let scale = Self::estimate_scale(residuals);
        RobustLoss::Huber(1.345 * scale)
    }

    /// Create a Cauchy loss function with an automatically estimated parameter.
    ///
    /// The c parameter is set to 2.385 times the estimated scale, which provides
    /// 95% efficiency for normally distributed errors.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The residuals used to estimate the scale
    pub fn auto_cauchy(residuals: &Array1<f64>) -> Self {
        let scale = Self::estimate_scale(residuals);
        RobustLoss::Cauchy(2.385 * scale)
    }

    /// Create a Tukey loss function with an automatically estimated parameter.
    ///
    /// The c parameter is set to 4.685 times the estimated scale, which provides
    /// 95% efficiency for normally distributed errors.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The residuals used to estimate the scale
    pub fn auto_tukey(residuals: &Array1<f64>) -> Self {
        let scale = Self::estimate_scale(residuals);
        RobustLoss::Tukey(4.685 * scale)
    }
}

/// Trait for problems that support robust fitting.
///
/// This trait extends the Problem trait with methods for handling robust fitting.
pub trait RobustProblem {
    /// Set the robust loss function to use.
    ///
    /// # Arguments
    ///
    /// * `loss` - The robust loss function to use
    fn set_loss_function(&mut self, loss: RobustLoss);

    /// Get the current robust loss function.
    fn loss_function(&self) -> &RobustLoss;

    /// Apply the robust loss function to compute weighted residuals.
    ///
    /// # Arguments
    ///
    /// * `residuals` - The original residuals
    ///
    /// # Returns
    ///
    /// The weighted residuals
    fn apply_robust_weights(&self, residuals: &Array1<f64>) -> Array1<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_least_squares_loss() {
        let loss = RobustLoss::LeastSquares;

        // Test with a residual of 2.0
        let (rho, weight) = loss.evaluate(2.0);
        assert_relative_eq!(rho, 4.0); // 2.0^2 = 4.0
        assert_relative_eq!(weight, 1.0); // Weight is always 1.0
    }

    #[test]
    fn test_huber_loss() {
        let delta = 1.5;
        let loss = RobustLoss::Huber(delta);

        // Test in the quadratic region
        let (rho_q, weight_q) = loss.evaluate(1.0);
        assert_relative_eq!(rho_q, 1.0); // 1.0^2 = 1.0
        assert_relative_eq!(weight_q, 1.0); // Weight is 1.0 in quadratic region

        // Test in the linear region
        let (rho_l, weight_l) = loss.evaluate(2.0);
        // rho = 2*delta*|z| - delta^2 = 2*1.5*2.0 - 1.5^2 = 6.0 - 2.25 = 3.75
        assert_relative_eq!(rho_l, 3.75);
        // weight = delta/|z| = 1.5/2.0 = 0.75
        assert_relative_eq!(weight_l, 0.75);
    }

    #[test]
    fn test_soft_l1_loss() {
        let loss = RobustLoss::SoftL1;

        // Test with various residuals
        let (rho1, weight1) = loss.evaluate(0.0);
        assert_relative_eq!(rho1, 0.0); // 2*(sqrt(1+0^2)-1) = 0
        assert_relative_eq!(weight1, 1.0); // 1/sqrt(1+0^2) = 1

        let (rho2, weight2) = loss.evaluate(1.0);
        assert_relative_eq!(rho2, 2.0 * (2.0_f64.sqrt() - 1.0)); // 2*(sqrt(2)-1)
        assert_relative_eq!(weight2, 1.0 / 2.0_f64.sqrt()); // 1/sqrt(2)
    }

    #[test]
    fn test_cauchy_loss() {
        let c = 2.5;
        let loss = RobustLoss::Cauchy(c);

        // Test with a residual of 1.0
        let (rho, weight) = loss.evaluate(1.0);
        // z_c = 1.0/2.5 = 0.4
        // rho = ln(1 + 0.4^2) = ln(1.16) ≈ 0.148
        assert_relative_eq!(rho, (1.0 + 0.4 * 0.4).ln(), epsilon = 1e-6);
        // weight = 1/(1 + 0.4^2) = 1/1.16 ≈ 0.862
        assert_relative_eq!(weight, 1.0 / (1.0 + 0.4 * 0.4), epsilon = 1e-6);
    }

    #[test]
    fn test_tukey_loss() {
        let c = 4.685;
        let loss = RobustLoss::Tukey(c);

        // Test with a residual well within the bound
        let (rho_in, weight_in) = loss.evaluate(1.0);
        // z_c = 1.0/4.685 ≈ 0.213
        // temp = 1 - z_c^2 ≈ 0.955
        // rho = c^2/6 * (1 - temp^3) ≈ 4.685^2/6 * (1 - 0.955^3) ≈ 3.66 * 0.131 ≈ 0.48
        let z_c = 1.0 / c;
        let temp = 1.0 - z_c * z_c;
        let temp_cubed = temp * temp * temp;
        let expected_rho = c * c / 6.0 * (1.0 - temp_cubed);
        assert_relative_eq!(rho_in, expected_rho, epsilon = 1e-6);
        // weight = temp^2 ≈ 0.955^2 ≈ 0.912
        assert_relative_eq!(weight_in, temp * temp, epsilon = 1e-6);

        // Test with a residual outside the bound
        let (rho_out, weight_out) = loss.evaluate(5.0);
        // Outside the bound, rho is constant and weight is 0
        assert_relative_eq!(rho_out, c * c / 6.0, epsilon = 1e-6);
        assert_relative_eq!(weight_out, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_custom_loss() {
        // Create a custom loss function: |z|^1.5
        let loss = RobustLoss::custom(|z| {
            let abs_z = z.abs();
            let rho = abs_z.powf(1.5);
            let weight = if z != 0.0 { abs_z.powf(-0.5) } else { 1.0 };
            (rho, weight)
        });

        // Test with a residual of 4.0
        let (rho, weight) = loss.evaluate(4.0);
        // rho = |4.0|^1.5 = 8.0
        assert_relative_eq!(rho, 8.0, epsilon = 1e-6);
        // weight = |4.0|^(-0.5) = 0.5
        assert_relative_eq!(weight, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_apply_to_residuals() {
        let residuals = array![1.0, -2.0, 3.0, -4.0];

        // Test with Huber loss, delta = 2.0
        let loss = RobustLoss::Huber(2.0);
        let (loss_sum, weights) = loss.apply(&residuals);

        // Expected weights: [1.0, 1.0, 2/3, 0.5]
        let expected_weights = array![1.0, 1.0, 2.0 / 3.0, 0.5];

        // Expected loss sum: 1^2 + 2^2 + (2*2*3 - 2^2) + (2*2*4 - 2^2) = 1 + 4 + 8 + 12 = 25
        assert_relative_eq!(loss_sum, 25.0, epsilon = 1e-6);

        for (w, ew) in weights.iter().zip(expected_weights.iter()) {
            assert_relative_eq!(w, ew, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_weighted_residuals() {
        let residuals = array![1.0, -2.0, 3.0, -4.0];

        // Test with Huber loss, delta = 2.0
        let loss = RobustLoss::Huber(2.0);
        let (weighted, loss_sum) = loss.weighted_residuals(&residuals);

        // Expected weighted residuals: [1.0, -2.0, 3.0*sqrt(2/3), -4.0*sqrt(0.5)]
        let expected_weighted = array![1.0, -2.0, 3.0 * (2.0 / 3.0).sqrt(), -4.0 * 0.5_f64.sqrt()];

        assert_relative_eq!(loss_sum, 25.0, epsilon = 1e-6);

        for (w, ew) in weighted.iter().zip(expected_weighted.iter()) {
            assert_relative_eq!(w, ew, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_estimate_scale() {
        // Test with a simple array with known median absolute deviation
        let residuals = array![1.0, -2.0, 3.0, -4.0, 5.0];

        // Absolute residuals: [1, 2, 3, 4, 5]
        // Median is 3, so MAD = 1 (median of |x - median| = [2, 1, 0, 1, 2])
        // Scaled MAD = 1 * 1.4826 ≈ 1.4826

        let scale = RobustLoss::estimate_scale(&residuals);
        assert_relative_eq!(scale, 1.4826, epsilon = 1e-4);

        // Test with normally distributed values
        let normal_residuals = array![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0];

        // For standard normal values, scale should be close to 1
        let normal_scale = RobustLoss::estimate_scale(&normal_residuals);
        assert!(
            normal_scale > 0.5 && normal_scale < 1.5,
            "Scale for normal data should be near 1, got {}",
            normal_scale
        );
    }

    #[test]
    fn test_auto_parameter_selection() {
        let residuals = array![1.0, -2.0, 3.0, -4.0, 5.0];
        let scale = RobustLoss::estimate_scale(&residuals);

        // Check that auto parameters use the recommended factors
        let huber = RobustLoss::auto_huber(&residuals);
        if let RobustLoss::Huber(delta) = huber {
            assert_relative_eq!(delta, 1.345 * scale, epsilon = 1e-6);
        } else {
            panic!("Expected Huber loss");
        }

        let cauchy = RobustLoss::auto_cauchy(&residuals);
        if let RobustLoss::Cauchy(c) = cauchy {
            assert_relative_eq!(c, 2.385 * scale, epsilon = 1e-6);
        } else {
            panic!("Expected Cauchy loss");
        }

        let tukey = RobustLoss::auto_tukey(&residuals);
        if let RobustLoss::Tukey(c) = tukey {
            assert_relative_eq!(c, 4.685 * scale, epsilon = 1e-6);
        } else {
            panic!("Expected Tukey loss");
        }
    }
}
