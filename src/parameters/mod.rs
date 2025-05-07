//! # Parameter System
//!
//! This module provides a comprehensive parameter system for optimization problems, similar to lmfit-py.
//! It allows named parameters with bounds, expressions, and constraints to be used in optimization problems.
//!
//! ## Key Features
//!
//! - **Named Parameters**: Work with descriptive parameter names rather than array indices
//! - **Bounds Constraints**: Easily define min/max bounds with automatic handling during optimization
//! - **Parameter Expressions**: Define parameters as mathematical functions of other parameters
//! - **Algebraic Constraints**: Create constraints between parameters using mathematical expressions
//! - **Serialization Support**: Save and load parameter collections with serde
//!
//! ## Core Components
//!
//! - [`Parameter`]: Individual parameters with values, bounds, and varying flags
//! - [`Parameters`]: A collection of parameters with dependency tracking and expression evaluation
//! - [`Bounds`] and [`BoundsTransform`]: Handle parameter bounds during optimization
//! - [`Expression`]: Parse and evaluate mathematical expressions for parameter constraints
//! - [`Constraint`]: Define relationships between parameters
//!
//! ## Example Usage
//!
//! ```rust
//! use lmopt_rs::parameters::{Parameter, Parameters};
//!
//! // Create a parameter collection
//! let mut params = Parameters::new();
//!
//! // Add simple parameters
//! params.add_param("linear", 1.0).unwrap();
//!
//! // Add parameters with bounds
//! params.add_param_with_bounds("amplitude", 3.0, 0.0, 10.0).unwrap();
//! params.add_param_with_bounds("decay", 0.5, 0.0, f64::INFINITY).unwrap();
//!
//! // Fix a parameter (won't be varied during optimization)
//! params.get_mut("linear").unwrap().set_vary(false).unwrap();
//!
//! // Add a parameter with an expression
//! params.add_param_with_expr("half_life", 1.386, "ln(2) / decay").unwrap();
//!
//! // Access parameter values
//! let amplitude = params.get("amplitude").unwrap().value();
//! let half_life = params.get("half_life").unwrap().value();
//!
//! // Convert to array for optimizer (only varying parameters)
//! let param_array = params.to_array().unwrap();
//!
//! // After optimization, update from array
//! // params.update_from_array(&optimized_array).unwrap();
//! ```
//!
//! For comprehensive documentation, see the [Parameter System guide](https://docs.rs/lmopt-rs/latest/lmopt_rs/docs/concepts/parameters.md).

pub mod bounds;
pub mod constraints;
pub mod expression;
pub mod parameter;
pub mod parameters;

// Include tests
#[cfg(test)]
mod tests;

// Re-export key types
pub use bounds::{Bounds, BoundsError, BoundsTransform};
pub use constraints::{Constraint, ConstraintError, ConstraintType, Constraints};
pub use expression::{EvaluationContext, Expression, ExpressionError, SimpleContext};
pub use parameter::{Parameter, ParameterError};
pub use parameters::{Parameters, SerializationError};
