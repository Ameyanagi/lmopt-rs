//! Main test file for lmopt-rs
//!
//! This file organizes and includes all test modules for the library.

// Utility tests
mod matrix_conversion;
mod matrix_operations;
mod matrix_properties;

// Core algorithm tests
mod lm_algorithm;
mod lm_optimization;
mod problem_trait;

// Parameter system tests
mod parameters;

// Uncertainty calculation tests
mod uncertainty;

// Integration tests that test the library as a whole
mod integration;

/// Test helpers - common utilities for tests
pub mod test_helpers {
    use ndarray::{Array1, Array2};
    use std::f64::EPSILON;

    /// Check if two f64 values are approximately equal
    pub fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Check if two arrays are approximately equal
    pub fn array_approx_eq(a: &Array1<f64>, b: &Array1<f64>, tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }

        for i in 0..a.len() {
            if !approx_eq(a[i], b[i], tol) {
                return false;
            }
        }

        true
    }

    /// Check if two matrices are approximately equal
    pub fn matrix_approx_eq(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }

        for i in 0..a.shape()[0] {
            for j in 0..a.shape()[1] {
                if !approx_eq(a[[i, j]], b[[i, j]], tol) {
                    return false;
                }
            }
        }

        true
    }

    /// Generate a linear test model: y = m*x + b
    pub struct LinearModel {
        x: Array1<f64>,
        y: Array1<f64>,
    }

    impl LinearModel {
        pub fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
            Self { x, y }
        }

        pub fn generate(m: f64, b: f64, n: usize) -> (Self, Array1<f64>) {
            let x = Array1::linspace(0.0, 10.0, n);
            let y = x.mapv(|x_val| m * x_val + b);
            let true_params = Array1::from_vec(vec![m, b]);

            (Self::new(x, y), true_params)
        }

        pub fn generate_with_noise(
            m: f64,
            b: f64,
            n: usize,
            noise_level: f64,
        ) -> (Self, Array1<f64>) {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            let x = Array1::linspace(0.0, 10.0, n);
            let y = x.mapv(|x_val| {
                let noise = rng.gen_range(-noise_level..noise_level);
                m * x_val + b + noise
            });
            let true_params = Array1::from_vec(vec![m, b]);

            (Self::new(x, y), true_params)
        }
    }

    impl lmopt_rs::problem::Problem for LinearModel {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>, lmopt_rs::error::LmOptError> {
            let m = params[0];
            let b = params[1];

            let predicted = self.x.mapv(|x_val| m * x_val + b);
            Ok(&predicted - &self.y)
        }

        fn parameter_count(&self) -> usize {
            2 // m and b
        }

        fn residual_count(&self) -> usize {
            self.x.len()
        }

        fn jacobian(
            &self,
            _params: &Array1<f64>,
        ) -> Result<Array2<f64>, lmopt_rs::error::LmOptError> {
            let n = self.x.len();
            let mut jacobian = Array2::<f64>::zeros((n, 2));

            for i in 0..n {
                // ∂f/∂m = x
                jacobian[[i, 0]] = self.x[i];

                // ∂f/∂b = 1
                jacobian[[i, 1]] = 1.0;
            }

            Ok(jacobian)
        }

        fn has_custom_jacobian(&self) -> bool {
            true
        }
    }
}
