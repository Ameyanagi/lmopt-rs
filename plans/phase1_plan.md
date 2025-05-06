# Phase 1 Implementation Plan: Core Infrastructure

This document outlines the detailed implementation plan for Phase 1 of the `lmopt-rs` project, focusing on building the core infrastructure necessary for the Levenberg-Marquardt algorithm.

## Goals for Phase 1

1. Set up the project structure and dependencies
2. Implement matrix conversion utilities between ndarray, faer, and nalgebra
3. Define the Problem trait compatible with the levenberg-marquardt crate
4. Implement the core Levenberg-Marquardt algorithm

## Detailed Tasks

### 1. Project Structure and Dependencies (Week 1, Days 1-2)

#### 1.1 Configure Cargo.toml with Dependencies

```toml
[package]
name = "lmopt-rs"
version = "0.1.0"
edition = "2021"
authors = ["Ameyanagi"]
description = "A Rust implementation of the Levenberg-Marquardt algorithm with uncertainty calculation"
license = "MIT"
repository = "https://github.com/ameyanagi/lmopt-rs"
readme = "README.md"
keywords = ["optimization", "levenberg-marquardt", "curve-fitting", "least-squares"]
categories = ["mathematics", "science", "algorithms"]

[dependencies]
# Linear algebra libraries
faer = "0.22"
faer-ext = { version = "0.6", features = ["ndarray"] }
ndarray = "0.15"
nalgebra = "0.32"  # For compatibility with levenberg-marquardt

# Error handling
thiserror = "1.0"

# Random number generation (for Monte Carlo methods)
rand = "0.8"

# Serialization/deserialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[dev-dependencies]
# Testing utilities
approx = "0.5"
criterion = "0.5"  # For benchmarking
levenberg-marquardt = "0.8"  # For compatibility testing

[features]
default = []
```

#### 1.2 Set Up Basic Module Structure (Week 1, Days 3-4)

Create the following files:

- `src/lib.rs` - Main library entry point
- `src/error.rs` - Error definitions
- `src/problem.rs` - Problem definition trait
- `src/lm.rs` - Main Levenberg-Marquardt algorithm
- `src/utils/mod.rs` - Utilities module
- `src/utils/matrix_convert.rs` - Matrix conversion utilities

#### 1.3 Implement Error Handling (Week 1, Days 4-5)

In `src/error.rs`:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LmOptError {
    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    #[error("Matrix conversion error: {0}")]
    ConversionError(String),
    
    #[error("Singular matrix encountered")]
    SingularMatrix,
    
    #[error("Algorithm failed to converge: {0}")]
    ConvergenceFailure(String),
    
    #[error("Invalid parameter value: {0}")]
    InvalidParameter(String),
    
    #[error("Bounds error: {0}")]
    BoundsError(String),
    
    #[error("Function evaluation error: {0}")]
    FunctionEvaluation(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, LmOptError>;
```

### 2. Matrix Conversion Utilities (Week 1, Day 5 - Week 2, Day 2)

#### 2.1 Implement Conversion Between ndarray and faer

In `src/utils/matrix_convert.rs`:

```rust
use faer::Mat;
use ndarray::{Array1, Array2};
use crate::error::{LmOptError, Result};

pub fn ndarray_to_faer<T: faer::Scalar>(arr: &Array2<T>) -> Result<Mat<T>> {
    // Implementation
}

pub fn faer_to_ndarray<T: faer::Scalar>(mat: &Mat<T>) -> Result<Array2<T>> {
    // Implementation
}

pub fn ndarray_vec_to_faer<T: faer::Scalar>(arr: &Array1<T>) -> Result<faer::Col<T>> {
    // Implementation
}

pub fn faer_vec_to_ndarray<T: faer::Scalar>(vec: &faer::Col<T>) -> Result<Array1<T>> {
    // Implementation
}
```

#### 2.2 Implement Conversion Between nalgebra and faer/ndarray

Add functions for nalgebra conversions in the same file:

```rust
pub fn nalgebra_to_faer<T: faer::Scalar>(mat: &nalgebra::DMatrix<T>) -> Result<Mat<T>> {
    // Implementation
}

pub fn faer_to_nalgebra<T: faer::Scalar>(mat: &Mat<T>) -> Result<nalgebra::DMatrix<T>> {
    // Implementation
}

pub fn nalgebra_to_ndarray<T: nalgebra::RealField>(mat: &nalgebra::DMatrix<T>) -> Result<Array2<T>> {
    // Implementation
}

pub fn ndarray_to_nalgebra<T: nalgebra::RealField>(arr: &Array2<T>) -> Result<nalgebra::DMatrix<T>> {
    // Implementation
}
```

#### 2.3 Add Unit Tests for Matrix Conversions

Add test module with comprehensive tests for all conversion functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_ndarray_faer_roundtrip() {
        // Test roundtrip conversion from ndarray to faer and back
    }
    
    #[test]
    fn test_nalgebra_faer_roundtrip() {
        // Test roundtrip conversion from nalgebra to faer and back
    }
    
    #[test]
    fn test_ndarray_nalgebra_roundtrip() {
        // Test roundtrip conversion from ndarray to nalgebra and back
    }
    
    #[test]
    fn test_matrix_dimensions() {
        // Test handling of different matrix dimensions
    }
    
    #[test]
    fn test_error_handling() {
        // Test error handling for invalid conversions
    }
}
```

### 3. Problem Definition (Week 2, Days 3-7)

#### 3.1 Define Problem Trait

In `src/problem.rs`:

```rust
use ndarray::{Array1, Array2};
use crate::error::Result;

/// A trait representing a nonlinear least squares problem.
pub trait Problem {
    /// Evaluate the residuals at the given parameters.
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>>;
    
    /// Get the number of parameters.
    fn parameter_count(&self) -> usize;
    
    /// Get the number of residuals.
    fn residual_count(&self) -> usize;
    
    /// Evaluate the Jacobian matrix.
    /// 
    /// Default implementation uses Rust's autodiff for automatic differentiation.
    /// If autodiff is not applicable, falls back to finite differences.
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        // Default implementation uses autodiff when possible, fallback to finite differences
        crate::utils::autodiff::jacobian(self, params)
    }
    
    /// Check if this problem provides a custom Jacobian implementation.
    fn has_custom_jacobian(&self) -> bool {
        false
    }
    
    /// Evaluate a scalar function (e.g., sum of squared residuals) at the given parameters.
    /// Useful for optimization and line searches.
    fn eval_scalar(&self, params: &Array1<f64>) -> Result<f64> {
        let residuals = self.eval(params)?;
        Ok(residuals.iter().map(|r| r.powi(2)).sum())
    }
}
```

#### 3.2 Implement Adapter for levenberg-marquardt Problem

Create an adapter to make our Problem trait compatible with the levenberg-marquardt crate:

```rust
use levenberg_marquardt::{LeastSquaresProblem, LeastSquaresFunc};
use nalgebra::{DMatrix, DVector};

/// Adapter to use lmopt-rs Problem with levenberg-marquardt crate.
pub struct LevenbergMarquardtAdapter<P: Problem> {
    problem: P,
}

impl<P: Problem> LevenbergMarquardtAdapter<P> {
    pub fn new(problem: P) -> Self {
        Self { problem }
    }
}

impl<P: Problem> LeastSquaresProblem<f64> for LevenbergMarquardtAdapter<P> {
    type Params = DVector<f64>;
    type Residuals = DVector<f64>;
    type Jacobian = DMatrix<f64>;
    
    fn residuals(&self, params: &Self::Params) -> Result<Self::Residuals, levenberg_marquardt::Error> {
        // Convert params to ndarray
        // Call problem.eval
        // Convert result to DVector
    }
    
    fn jacobian(&self, params: &Self::Params) -> Result<Self::Jacobian, levenberg_marquardt::Error> {
        // If problem has analytical jacobian, use it
        // Otherwise, use numerical differentiation
    }
    
    fn params_len(&self) -> usize {
        self.problem.parameter_count()
    }
    
    fn residuals_len(&self) -> usize {
        self.problem.residual_count()
    }
}
```

#### 3.3 Add Unit Tests for Problem Trait

Create test module with tests for the Problem trait and adapter:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    
    // Define a simple test problem (e.g., y = a * exp(-b * x))
    struct ExponentialProblem {
        x: Array1<f64>,
        y: Array1<f64>,
    }
    
    impl Problem for ExponentialProblem {
        // Implement required methods
    }
    
    #[test]
    fn test_problem_eval() {
        // Test basic problem evaluation
    }
    
    #[test]
    fn test_problem_jacobian() {
        // Test jacobian calculation
    }
    
    #[test]
    fn test_levenberg_marquardt_adapter() {
        // Test adapter with levenberg-marquardt crate
    }
}
```

### 4. Core LM Algorithm (Week 3, Days 1-7)

#### 4.1 Implement Basic LM Algorithm Structure

In `src/lm.rs`:

```rust
use faer::{Mat, Col};
use ndarray::{Array1, Array2};
use crate::error::{LmOptError, Result};
use crate::problem::Problem;
use crate::utils::matrix_convert::{ndarray_to_faer, faer_to_ndarray};

/// Configuration options for the Levenberg-Marquardt algorithm.
pub struct LmConfig {
    pub max_iterations: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    pub initial_lambda: f64,
    pub lambda_factor: f64,
    pub min_lambda: f64,
    pub max_lambda: f64,
}

impl Default for LmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-8,
            initial_lambda: 1e-3,
            lambda_factor: 10.0,
            min_lambda: 1e-10,
            max_lambda: 1e10,
        }
    }
}

/// Result of the Levenberg-Marquardt optimization.
pub struct LmResult {
    pub params: Array1<f64>,
    pub residuals: Array1<f64>,
    pub cost: f64,
    pub iterations: usize,
    pub success: bool,
    pub message: String,
    pub jacobian: Option<Array2<f64>>,
}

/// The Levenberg-Marquardt optimizer.
pub struct LevenbergMarquardt {
    config: LmConfig,
}

impl LevenbergMarquardt {
    pub fn new(config: LmConfig) -> Self {
        Self { config }
    }
    
    pub fn minimize<P: Problem>(&self, problem: &P, initial_params: Array1<f64>) -> Result<LmResult> {
        // Implement the LM algorithm
    }
    
    fn calculate_step(&self, j: &Mat<f64>, r: &Col<f64>, lambda: f64) -> Result<Col<f64>> {
        // Calculate the LM step using (J^T J + λI)^-1 J^T r
    }
    
    fn check_convergence(
        &self,
        params: &Array1<f64>,
        new_params: &Array1<f64>,
        residuals: &Array1<f64>,
        new_residuals: &Array1<f64>,
        gradient_norm: f64,
    ) -> bool {
        // Check convergence criteria
    }
}
```

#### 4.2 Add Trust Region Implementation

In `src/lm/trust_region.rs`:

```rust
use faer::{Mat, Col};
use crate::error::{LmOptError, Result};

/// Calculate the Levenberg-Marquardt step using trust region approach.
pub fn calculate_step(j: &Mat<f64>, r: &Col<f64>, lambda: f64) -> Result<Col<f64>> {
    // Implement trust region approach
}
```

#### 4.3 Implement Convergence Criteria

In `src/lm/convergence.rs`:

```rust
use ndarray::Array1;

/// Check if the algorithm has converged based on parameter, residual, and gradient changes.
pub fn check_convergence(
    params: &Array1<f64>,
    new_params: &Array1<f64>,
    residuals: &Array1<f64>,
    new_residuals: &Array1<f64>,
    gradient_norm: f64,
    ftol: f64,
    xtol: f64,
    gtol: f64,
) -> bool {
    // Implement convergence checks
}
```

#### 4.4 Add Step Calculation with Damping

In `src/lm/step.rs`:

```rust
use faer::{Mat, Col};
use crate::error::Result;

/// Calculate the step using the Levenberg-Marquardt formula.
pub fn calculate_lm_step(j: &Mat<f64>, r: &Col<f64>, lambda: f64) -> Result<Col<f64>> {
    // Implement LM step calculation
}

/// Update the damping parameter based on success of the step.
pub fn update_lambda(
    lambda: f64,
    success: bool,
    lambda_factor: f64,
    min_lambda: f64,
    max_lambda: f64,
) -> f64 {
    // Update lambda based on step success
}
```

#### 4.5 Add Unit Tests for LM Algorithm

Implement comprehensive tests for the LM algorithm:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use crate::problem::Problem;
    
    // Define test problems
    
    #[test]
    fn test_lm_linear_problem() {
        // Test with a simple linear problem
    }
    
    #[test]
    fn test_lm_nonlinear_problem() {
        // Test with a nonlinear problem
    }
    
    #[test]
    fn test_lm_step_calculation() {
        // Test step calculation
    }
    
    #[test]
    fn test_lambda_updating() {
        // Test lambda update logic
    }
    
    #[test]
    fn test_convergence_criteria() {
        // Test convergence checking
    }
}
```

### 5. Integration Testing (Week 4, Days 1-5)

#### 5.1 Create Integration Tests

In `tests/matrix_conversion.rs`:
```rust
// Test matrix conversion utilities
```

In `tests/problem_definition.rs`:
```rust
// Test Problem trait and adapter
```

In `tests/lm_algorithm.rs`:
```rust
// Test LM algorithm implementation
```

#### 5.2 Benchmark Against Reference Implementations

In `benches/lm_algorithm.rs`:
```rust
// Benchmark lmopt-rs against levenberg-marquardt
```

## Acceptance Criteria for Phase 1

1. **Matrix Conversion Utilities**:
   - All conversion functions are implemented and tested
   - Roundtrip conversions preserve values within floating-point precision
   - Error handling is comprehensive and informative

2. **Problem Definition**:
   - Problem trait is defined and documented
   - Adapter for levenberg-marquardt compatibility works correctly
   - Interface is intuitive and follows Rust idioms

3. **Core LM Algorithm**:
   - Algorithm correctly implements the Levenberg-Marquardt method
   - Trust region approach is robust
   - Convergence criteria work as expected
   - Results match or exceed reference implementations
   - Performance is acceptable

4. **Documentation and Testing**:
   - Code is well-documented with docstrings
   - Unit tests cover all main functionality
   - Integration tests verify compatibility and correctness
   - Benchmarks compare performance with reference implementations

## Dependencies

- Rust stable toolchain
- faer 0.22
- faer-ext 0.6 with ndarray feature
- ndarray 0.15
- nalgebra 0.32
- thiserror 1.0
- levenberg-marquardt 0.8 (for testing)