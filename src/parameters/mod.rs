//! Parameter system implementation
//!
//! This module provides a parameter system similar to lmfit-py, with support for:
//! - Named parameters with values and uncertainties
//! - Bounds constraints (min/max) with automatic handling for optimizers
//! - Parameter expressions for algebraic constraints
//! - Serialization/deserialization support
//!
//! The parameter system is built around three main components:
//! - `Parameter`: Individual parameters with values, bounds, expressions
//! - `Parameters`: A collection of parameters with dependency tracking
//! - `Bounds` and `BoundsTransform`: Handling of bounds constraints

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
