//! # lmopt-rs
//!
//! `lmopt-rs` is a Rust implementation of the Levenberg-Marquardt algorithm
//! for nonlinear least-squares optimization, with uncertainty calculation capabilities.
//!
//! The library provides:
//! - A Levenberg-Marquardt implementation compatible with the `levenberg-marquardt` crate
//! - Uncertainty calculation features similar to those in `lmfit-py`
//! - A flexible parameter system with bounds and constraints
//! - A model system for common fitting problems
//!
//! ## Basic Usage
//!
//! ```
//! // To be implemented
//! ```

// Public modules
pub mod error;

// Parameter system
pub mod parameters;

// Conditional modules
#[cfg(feature = "matrix")]
mod utils;

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
mod models;

// Re-exports for convenience
pub use error::{LmOptError, Result};

#[cfg(feature = "lm")]
pub use lm::LevenbergMarquardt;

#[cfg(feature = "lm")]
pub use problem::Problem;

#[cfg(feature = "lm")]
pub use problem_params::ParameterProblem;

#[cfg(feature = "lm")]
pub use uncertainty::{
    UncertaintyCalculator, 
    UncertaintyResult, 
    ConfidenceInterval,
    MonteCarloResult,
    covariance_matrix, 
    standard_errors,
    uncertainty_analysis,
    uncertainty_analysis_with_monte_carlo,
    monte_carlo_covariance,
    monte_carlo_refit,
    propagate_uncertainty,
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