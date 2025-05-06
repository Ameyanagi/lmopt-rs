//! Benchmarks comparing lmopt-rs with reference implementations
//!
//! This file contains benchmarks that compare the performance of lmopt-rs
//! with levenberg-marquardt and (indirectly) lmfit-py.

#![feature(test)]
extern crate test;

use lmopt_rs::{
    lm::{LMConfig, LevenbergMarquardt},
    problem::Problem,
};
use ndarray::{Array1, Array2};
use test::Bencher;

// Import levenberg-marquardt for comparison
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt as LM};
use nalgebra::{DMatrix, DVector};

/// Linear model for lmopt-rs
struct LinearModel {
    x: Array1<f64>,
    y: Array1<f64>,
}

impl LinearModel {
    fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self { x, y }
    }

    fn generate(m: f64, b: f64, n: usize) -> Self {
        let x = Array1::linspace(0.0, 10.0, n);
        let y = x.mapv(|x_val| m * x_val + b);
        Self::new(x, y)
    }
}

impl Problem for LinearModel {
    type Scalar = f64;
    type Parameters = Array1<f64>;
    type ResidualVector = Array1<f64>;
    type JacobianMatrix = Array2<f64>;

    fn residuals(&self, params: &Self::Parameters) -> Self::ResidualVector {
        let m = params[0];
        let b = params[1];

        let predicted = self.x.mapv(|x_val| m * x_val + b);
        &predicted - &self.y
    }

    fn jacobian(&self, _params: &Self::Parameters) -> Self::JacobianMatrix {
        let n = self.x.len();
        let mut jacobian = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            // ∂f/∂m = x
            jacobian[[i, 0]] = self.x[i];

            // ∂f/∂b = 1
            jacobian[[i, 1]] = 1.0;
        }

        jacobian
    }

    fn num_residuals(&self) -> usize {
        self.x.len()
    }

    fn num_parameters(&self) -> usize {
        2 // m and b
    }
}

/// Linear model for levenberg-marquardt crate
struct LMLinearModel {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LMLinearModel {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self { x, y }
    }

    fn generate(m: f64, b: f64, n: usize) -> Self {
        let x: Vec<f64> = (0..n)
            .map(|i| 10.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let y: Vec<f64> = x.iter().map(|&x_val| m * x_val + b).collect();
        Self::new(x, y)
    }
}

impl LeastSquaresProblem<f64, DVector<f64>, DMatrix<f64>> for LMLinearModel {
    fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
        let m = params[0];
        let b = params[1];

        let residuals: Vec<f64> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(&x_val, &y_val)| m * x_val + b - y_val)
            .collect();

        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self, _params: &DVector<f64>) -> Option<DMatrix<f64>> {
        let n = self.x.len();
        let mut jacobian = DMatrix::zeros(n, 2);

        for i in 0..n {
            // ∂f/∂m = x
            jacobian[(i, 0)] = self.x[i];

            // ∂f/∂b = 1
            jacobian[(i, 1)] = 1.0;
        }

        Some(jacobian)
    }
}

/// Benchmark lmopt-rs linear model
#[bench]
fn bench_lmopt_linear_1000(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 1000);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);

    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-8)
        .function_tolerance(1e-8);

    let lm = LevenbergMarquardt::new(config);

    b.iter(|| test::black_box(lm.minimize(&model, initial_params.clone())));
}

/// Benchmark levenberg-marquardt crate linear model
#[bench]
fn bench_lm_crate_linear_1000(b: &mut Bencher) {
    let model = LMLinearModel::generate(2.5, -3.0, 1000);
    let initial_params = DVector::from_vec(vec![1.0, 0.0]);

    b.iter(|| test::black_box(LM::new().minimize(&model, &initial_params)));
}

/// Exponential model for lmopt-rs
struct ExponentialModel {
    x: Array1<f64>,
    y: Array1<f64>,
}

impl ExponentialModel {
    fn new(x: Array1<f64>, y: Array1<f64>) -> Self {
        Self { x, y }
    }

    fn generate(a: f64, b: f64, n: usize) -> Self {
        let x = Array1::linspace(0.0, 2.0, n);
        let y = x.mapv(|x_val| a * (b * x_val).exp());
        Self::new(x, y)
    }
}

impl Problem for ExponentialModel {
    type Scalar = f64;
    type Parameters = Array1<f64>;
    type ResidualVector = Array1<f64>;
    type JacobianMatrix = Array2<f64>;

    fn residuals(&self, params: &Self::Parameters) -> Self::ResidualVector {
        let a = params[0];
        let b = params[1];

        let predicted = self.x.mapv(|x_val| a * (b * x_val).exp());
        &predicted - &self.y
    }

    fn jacobian(&self, params: &Self::Parameters) -> Self::JacobianMatrix {
        let a = params[0];
        let b = params[1];
        let n = self.x.len();

        let mut jacobian = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let x_val = self.x[i];
            let exp_term = (b * x_val).exp();

            // ∂f/∂a = exp(b * x)
            jacobian[[i, 0]] = exp_term;

            // ∂f/∂b = a * x * exp(b * x)
            jacobian[[i, 1]] = a * x_val * exp_term;
        }

        jacobian
    }

    fn num_residuals(&self) -> usize {
        self.x.len()
    }

    fn num_parameters(&self) -> usize {
        2 // a and b
    }
}

/// Exponential model for levenberg-marquardt crate
struct LMExponentialModel {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LMExponentialModel {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Self { x, y }
    }

    fn generate(a: f64, b: f64, n: usize) -> Self {
        let x: Vec<f64> = (0..n)
            .map(|i| 2.0 * (i as f64) / (n as f64 - 1.0))
            .collect();
        let y: Vec<f64> = x.iter().map(|&x_val| a * (b * x_val).exp()).collect();
        Self::new(x, y)
    }
}

impl LeastSquaresProblem<f64, DVector<f64>, DMatrix<f64>> for LMExponentialModel {
    fn residuals(&self, params: &DVector<f64>) -> Option<DVector<f64>> {
        let a = params[0];
        let b = params[1];

        let residuals: Vec<f64> = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(&x_val, &y_val)| a * (b * x_val).exp() - y_val)
            .collect();

        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self, params: &DVector<f64>) -> Option<DMatrix<f64>> {
        let a = params[0];
        let b = params[1];
        let n = self.x.len();

        let mut jacobian = DMatrix::zeros(n, 2);

        for i in 0..n {
            let x_val = self.x[i];
            let exp_term = (b * x_val).exp();

            // ∂f/∂a = exp(b * x)
            jacobian[(i, 0)] = exp_term;

            // ∂f/∂b = a * x * exp(b * x)
            jacobian[(i, 1)] = a * x_val * exp_term;
        }

        Some(jacobian)
    }
}

/// Benchmark lmopt-rs exponential model
#[bench]
fn bench_lmopt_exponential_1000(b: &mut Bencher) {
    let model = ExponentialModel::generate(1.5, 0.5, 1000);
    let initial_params = Array1::from_vec(vec![1.0, 0.1]);

    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-8)
        .function_tolerance(1e-8);

    let lm = LevenbergMarquardt::new(config);

    b.iter(|| test::black_box(lm.minimize(&model, initial_params.clone())));
}

/// Benchmark levenberg-marquardt crate exponential model
#[bench]
fn bench_lm_crate_exponential_1000(b: &mut Bencher) {
    let model = LMExponentialModel::generate(1.5, 0.5, 1000);
    let initial_params = DVector::from_vec(vec![1.0, 0.1]);

    b.iter(|| test::black_box(LM::new().minimize(&model, &initial_params)));
}

// Benchmark scaling behavior across different problem sizes

/// Benchmark lmopt-rs with increasing dataset sizes
#[bench]
fn bench_lmopt_scaling_100(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 100);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    let lm = LevenbergMarquardt::new(LMConfig::default());

    b.iter(|| test::black_box(lm.minimize(&model, initial_params.clone())));
}

#[bench]
fn bench_lmopt_scaling_1000(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 1000);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    let lm = LevenbergMarquardt::new(LMConfig::default());

    b.iter(|| test::black_box(lm.minimize(&model, initial_params.clone())));
}

#[bench]
fn bench_lmopt_scaling_10000(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 10000);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    let lm = LevenbergMarquardt::new(LMConfig::default());

    b.iter(|| test::black_box(lm.minimize(&model, initial_params.clone())));
}

/// Benchmark levenberg-marquardt crate with increasing dataset sizes
#[bench]
fn bench_lm_crate_scaling_100(b: &mut Bencher) {
    let model = LMLinearModel::generate(2.5, -3.0, 100);
    let initial_params = DVector::from_vec(vec![1.0, 0.0]);

    b.iter(|| test::black_box(LM::new().minimize(&model, &initial_params)));
}

#[bench]
fn bench_lm_crate_scaling_1000(b: &mut Bencher) {
    let model = LMLinearModel::generate(2.5, -3.0, 1000);
    let initial_params = DVector::from_vec(vec![1.0, 0.0]);

    b.iter(|| test::black_box(LM::new().minimize(&model, &initial_params)));
}

#[bench]
fn bench_lm_crate_scaling_10000(b: &mut Bencher) {
    let model = LMLinearModel::generate(2.5, -3.0, 10000);
    let initial_params = DVector::from_vec(vec![1.0, 0.0]);

    b.iter(|| test::black_box(LM::new().minimize(&model, &initial_params)));
}
