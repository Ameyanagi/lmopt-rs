//! # lmopt-rs
//!
//! `lmopt-rs` is a Rust implementation of the Levenberg-Marquardt algorithm
//! for nonlinear least-squares optimization, with uncertainty calculation capabilities.
//!
//! ## Overview
//!
//! The library provides:
//! - A Levenberg-Marquardt implementation compatible with the `levenberg-marquardt` crate
//! - Uncertainty calculation features similar to those in `lmfit-py`
//! - A flexible parameter system with bounds and constraints
//! - A model system for common fitting problems
//! - Global optimization methods for finding global minima
//!
//! ## Documentation
//!
//! - [Getting Started](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/getting_started.md)
//! - Concepts:
//!   - [Parameter System](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/parameters.md)
//!   - [Model System](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/models.md)
//!   - [Levenberg-Marquardt Algorithm](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/lm_algorithm.md)
//!   - [Uncertainty Analysis](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/uncertainty.md)
//!   - [Global Optimization](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/global_optimization.md)
//! - Examples:
//!   - [Basic Fitting](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/examples/basic_fitting.md)
//!
//! ## Basic Usage
//!
//! ```rust
//! use lmopt_rs::{LevenbergMarquardt, Problem};
//! use ndarray::{array, Array1};
//!
//! // Define a problem that implements the Problem trait
//! struct QuadraticProblem;
//!
//! impl Problem for QuadraticProblem {
//!     fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
//!         // params[0] = a, params[1] = b, params[2] = c
//!         // We're fitting: y = a*x^2 + b*x + c
//!         
//!         // Sample data points for (x, y): (1, 6), (2, 11), (3, 18)
//!         let x_values = array![1.0, 2.0, 3.0];
//!         let y_values = array![6.0, 11.0, 18.0];
//!         
//!         // Calculate residuals: y_model - y_data
//!         let residuals = x_values.mapv(|x| {
//!             params[0] * x.powi(2) + params[1] * x + params[2]
//!         }) - y_values;
//!         
//!         Ok(residuals)
//!     }
//!     
//!     fn parameter_count(&self) -> usize {
//!         3  // a, b, c
//!     }
//!     
//!     fn residual_count(&self) -> usize {
//!         3  // Number of data points
//!     }
//! }
//!
//! fn main() -> lmopt_rs::Result<()> {
//!     // Create the problem
//!     let problem = QuadraticProblem;
//!     
//!     // Create the optimizer
//!     let mut optimizer = LevenbergMarquardt::with_default_config();
//!     
//!     // Initial guess for parameters [a, b, c]
//!     let initial_params = array![1.0, 1.0, 1.0];
//!     
//!     // Run the optimization
//!     let result = optimizer.minimize(&problem, initial_params)?;
//!     
//!     println!("Optimization successful: {}", result.success);
//!     println!("Final parameters: {:?}", result.params);
//!     println!("Cost (sum of squared residuals): {}", result.cost);
//!     println!("Number of iterations: {}", result.iterations);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ## Using the Parameter System
//!
//! For more details, see the [Parameter System documentation](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/parameters.md).
//!
//! ```rust
//! use lmopt_rs::parameters::{Parameter, Parameters};
//! use lmopt_rs::problem_params::ParameterProblem;
//!
//! // Create a parameter collection
//! let mut params = Parameters::new();
//!
//! // Add parameters with bounds
//! params.add_param_with_bounds("amplitude", 2.0, 0.0, 10.0)?;  // Between 0 and 10
//! params.add_param("center", 0.0)?;  // Unbounded
//! params.add_param_with_bounds("sigma", 1.0, 0.01, f64::INFINITY)?;  // Greater than 0.01
//!
//! // Fix a parameter (won't be varied during optimization)
//! params.get_mut("center").unwrap().set_vary(false)?;
//!
//! // Add a derived parameter
//! params.add_param_with_expr("fwhm", 2.355, "2.355 * sigma")?;
//! ```
//!
//! ## Using Built-in Models
//!
//! For more details, see the [Model System documentation](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/models.md).
//!
//! ```rust
//! use lmopt_rs::model::{fit, Model};
//! use lmopt_rs::models::{GaussianModel, ExponentialModel, add};
//! use ndarray::Array1;
//!
//! // Create a Gaussian model
//! let mut model = GaussianModel::new("", true);  // Empty prefix, with baseline
//!
//! // Create x and y data arrays
//! let x = Array1::linspace(-5.0, 5.0, 100);
//! let y = x.mapv(|x_val| {
//!     // Gaussian with amplitude=3.0, center=0.0, sigma=1.0, baseline=0.5
//!     3.0 * (-x_val.powi(2) / 2.0).exp() + 0.5
//! });
//!
//! // Fit the model to the data
//! let result = fit(&mut model, x.clone(), y.clone())?;
//!
//! // Print results
//! println!("Fit success: {}", result.success);
//! println!("Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
//! println!("Center: {:.4}", model.parameters().get("center").unwrap().value());
//! println!("Sigma: {:.4}", model.parameters().get("sigma").unwrap().value());
//! println!("Baseline: {:.4}", model.parameters().get("baseline").unwrap().value());
//! ```
//!
//! ## Uncertainty Analysis
//!
//! For more details, see the [Uncertainty Analysis documentation](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/uncertainty.md).
//!
//! ```rust
//! use lmopt_rs::uncertainty::uncertainty_analysis;
//!
//! // Calculate uncertainty after fitting
//! let uncertainty = uncertainty_analysis(&model, &result)?;
//!
//! // Print confidence intervals
//! for (name, interval) in &uncertainty.confidence_intervals {
//!     println!("{}: {:.4} ± {:.4} (95% CI: [{:.4}, {:.4}])",
//!              name, interval.value, interval.error,
//!              interval.lower, interval.upper);
//! }
//! ```
//!
//! ## Global Optimization
//!
//! For more details, see the [Global Optimization documentation](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/global_optimization.md).
//!
//! ```rust
//! use lmopt_rs::{optimize_global, DifferentialEvolution, Problem};
//!
//! // Define bounds for parameters
//! let bounds = vec![(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)];
//!
//! // Create a global optimizer
//! let optimizer = DifferentialEvolution::new();
//!
//! // Run global optimization
//! let result = optimizer.optimize(
//!     &problem,    // Problem to solve
//!     &bounds,     // Parameter bounds
//!     100,         // Number of iterations
//!     20,          // Population size
//!     1e-6         // Tolerance
//! )?;
//!
//! println!("Global optimization result: {:?}", result.params);
//! ```

// Public modules
pub mod error;

// Parameter system
pub mod parameters;

// Conditional modules
#[cfg(feature = "matrix")]
pub mod utils;

#[cfg(feature = "lm")]
pub mod problem;

#[cfg(feature = "lm")]
pub mod problem_params;

#[cfg(feature = "lm")]
pub mod lm;

#[cfg(feature = "lm")]
pub mod uncertainty;

#[cfg(feature = "lm")]
pub mod model;

#[cfg(feature = "lm")]
pub mod models;

// Re-exports for convenience
pub use error::{LmOptError, Result};

#[cfg(feature = "lm")]
pub use lm::{LevenbergMarquardt, ParallelLevenbergMarquardt};

#[cfg(feature = "lm")]
pub use problem::Problem;

#[cfg(feature = "lm")]
pub use problem_params::ParameterProblem;

#[cfg(feature = "lm")]
pub use uncertainty::{
    covariance_matrix, monte_carlo_covariance, monte_carlo_refit, propagate_uncertainty,
    standard_errors, uncertainty_analysis, uncertainty_analysis_with_monte_carlo,
    ConfidenceInterval, MonteCarloResult, UncertaintyCalculator, UncertaintyResult,
};

#[cfg(feature = "lm")]
pub mod global_opt;

#[cfg(feature = "lm")]
pub use global_opt::{
    optimize_global, optimize_global_parallel, optimize_global_param_problem, BasinHopping,
    DifferentialEvolution, GlobalOptResult, GlobalOptimizer, HybridGlobal,
    ParallelDifferentialEvolution, SimulatedAnnealing,
};

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
