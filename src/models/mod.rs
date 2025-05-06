//! Built-in model implementations for common fitting problems.
//!
//! This module provides a collection of commonly used model functions
//! for fitting data, such as Gaussian, Lorentzian, exponential, and
//! polynomial models.

use crate::error::{LmOptError, Result};
use crate::model::{BaseModel, Model};
use crate::parameters::{Parameter, Parameters};
use ndarray::{Array1, Array2};

mod composite;
mod exponential;
mod peak;
mod polynomial;
mod step;

// Re-export the models
// These models are not yet implemented
//pub use peak::{GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel};
pub use composite::{add, composite_with_shared_params, multiply, CompositeModel, Operation};
pub use exponential::{ExponentialModel, PowerLawModel};
pub use polynomial::{ConstantModel, LinearModel, PolynomialModel, QuadraticModel};
pub use step::{RectangleModel, SigmoidModel, StepModel};

/// Create a linear model with the specified parameter names
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names (e.g., "m" and "b" for "m:x" and "b")
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A linear model (y = m*x + b)
pub fn linear_model(prefix: &str, with_init: bool) -> LinearModel {
    LinearModel::new(prefix, with_init)
}

/// Create a quadratic model with the specified parameter names
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names (e.g., "a" for "a:x^2", "a:x", "a:c")
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A quadratic model (y = a*x^2 + b*x + c)
pub fn quadratic_model(prefix: &str, with_init: bool) -> QuadraticModel {
    QuadraticModel::new(prefix, with_init)
}

// These models are not yet implemented
/*
/// Create a Gaussian peak model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A Gaussian model (y = amplitude * exp(-(x-center)^2 / (2*sigma^2)) + baseline)
pub fn gaussian_model(prefix: &str, with_init: bool) -> GaussianModel {
    GaussianModel::new(prefix, with_init)
}

/// Create a Lorentzian peak model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A Lorentzian model (y = amplitude * gamma^2 / ((x-center)^2 + gamma^2) + baseline)
pub fn lorentzian_model(prefix: &str, with_init: bool) -> LorentzianModel {
    LorentzianModel::new(prefix, with_init)
}
*/

/// Create an exponential model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * An exponential model (y = amplitude * exp(-x/decay) + baseline)
pub fn exponential_model(prefix: &str, with_init: bool) -> ExponentialModel {
    ExponentialModel::new(prefix, with_init)
}

/// Create a power law model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A power law model (y = amplitude * x^exponent + baseline)
pub fn power_law_model(prefix: &str, with_init: bool) -> PowerLawModel {
    PowerLawModel::new(prefix, with_init)
}

/// Create a polynomial model of the specified degree
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `degree` - The degree of the polynomial
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A polynomial model of the specified degree
pub fn polynomial_model(prefix: &str, degree: usize, with_init: bool) -> PolynomialModel {
    PolynomialModel::new(prefix, degree, with_init)
}

/// Create a step model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A step model (y = amplitude * (x > center) + baseline)
pub fn step_model(prefix: &str, with_init: bool) -> StepModel {
    StepModel::new(prefix, with_init)
}

/// Create a sigmoid model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A sigmoid model (y = amplitude / (1 + exp(-(x-center)/sigma)) + baseline)
pub fn sigmoid_model(prefix: &str, with_init: bool) -> SigmoidModel {
    SigmoidModel::new(prefix, with_init)
}

/// Create a rectangle model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_init` - Whether to initialize parameters with reasonable values based on data
///
/// # Returns
///
/// * A rectangle model (y = amplitude * (x > center1 && x < center2) + baseline)
pub fn rectangle_model(prefix: &str, with_init: bool) -> RectangleModel {
    RectangleModel::new(prefix, with_init)
}

// Composite models - placeholder for future implementation
