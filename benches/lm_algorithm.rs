//! Benchmarks for the Levenberg-Marquardt algorithm
//!
//! This file contains benchmarks for the core LM algorithm to evaluate
//! performance and facilitate optimizations.

#![feature(test)]
extern crate test;

use lmopt_rs::{
    lm::{LevenbergMarquardt, LMConfig},
    problem::Problem,
};
use ndarray::{Array1, Array2};
use test::Bencher;

/// Simple linear model for benchmarking: y = m*x + b
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

/// Exponential model for benchmarking: y = a * exp(b * x)
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

// Rosenbrock function model for benchmarking:
// f(x,y) = (a - x)² + b(y - x²)²
// Residuals: [a - x, sqrt(b)*(y - x²)]
struct RosenbrockModel {
    // No data needed, just constants a and b
    a: f64,
    b: f64,
}

impl RosenbrockModel {
    fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }
}

impl Problem for RosenbrockModel {
    type Scalar = f64;
    type Parameters = Array1<f64>;
    type ResidualVector = Array1<f64>;
    type JacobianMatrix = Array2<f64>;
    
    fn residuals(&self, params: &Self::Parameters) -> Self::ResidualVector {
        let x = params[0];
        let y = params[1];
        
        Array1::from_vec(vec![
            self.a - x,
            (self.b).sqrt() * (y - x * x)
        ])
    }
    
    fn jacobian(&self, params: &Self::Parameters) -> Self::JacobianMatrix {
        let x = params[0];
        let _y = params[1];
        let sqrt_b = (self.b).sqrt();
        
        Array2::from_shape_vec((2, 2), vec![
            -1.0, 0.0,
            -2.0 * sqrt_b * x, sqrt_b
        ]).unwrap()
    }
    
    fn num_residuals(&self) -> usize {
        2
    }
    
    fn num_parameters(&self) -> usize {
        2 // x and y
    }
}

/// Benchmark linear model fitting with 1000 data points
#[bench]
fn bench_linear_model_1000(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 1000);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    
    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-8)
        .function_tolerance(1e-8);
    
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}

/// Benchmark exponential model fitting with 1000 data points
#[bench]
fn bench_exponential_model_1000(b: &mut Bencher) {
    let model = ExponentialModel::generate(1.5, 0.5, 1000);
    let initial_params = Array1::from_vec(vec![1.0, 0.1]);
    
    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-8)
        .function_tolerance(1e-8);
    
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}

/// Benchmark Rosenbrock function optimization
#[bench]
fn bench_rosenbrock(b: &mut Bencher) {
    let model = RosenbrockModel::new(1.0, 100.0);
    let initial_params = Array1::from_vec(vec![-1.2, 1.0]);
    
    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-8)
        .function_tolerance(1e-8);
    
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}

/// Benchmark linear model with varying number of data points
#[bench]
fn bench_linear_model_10(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 10);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    
    let config = LMConfig::default().max_iterations(100);
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}

#[bench]
fn bench_linear_model_100(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 100);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    
    let config = LMConfig::default().max_iterations(100);
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}

#[bench]
fn bench_linear_model_10000(b: &mut Bencher) {
    let model = LinearModel::generate(2.5, -3.0, 10000);
    let initial_params = Array1::from_vec(vec![1.0, 0.0]);
    
    let config = LMConfig::default().max_iterations(100);
    let lm = LevenbergMarquardt::new(config);
    
    b.iter(|| {
        test::black_box(lm.minimize(&model, &initial_params))
    });
}