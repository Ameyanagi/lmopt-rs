//! Levenberg-Marquardt algorithm implementation.
//!
//! This module provides an implementation of the Levenberg-Marquardt algorithm
//! for nonlinear least-squares optimization. The implementation is designed to be
//! efficient, robust, and configurable.
//!
//! For comprehensive documentation, see the [Levenberg-Marquardt Algorithm documentation](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/lm_algorithm.md).

// Include module declarations and re-exports
pub mod algorithm;
pub mod config;
pub mod convergence;
pub mod parallel;
pub mod robust;
pub mod step;
pub mod trust_region;

// Re-export key types
pub use algorithm::{LevenbergMarquardt, LmResult};
pub use config::{DecompositionMethod, DiffMethod, LmConfig};
pub use convergence::{ConvergenceCriteria, ConvergenceStatus};
pub use parallel::ParallelLevenbergMarquardt;
pub use robust::RobustLoss;
pub use step::{LmStep, StepResult};
pub use trust_region::TrustRegion;
