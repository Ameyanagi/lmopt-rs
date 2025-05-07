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
pub use composite::{add, composite_with_shared_params, multiply, CompositeModel, Operation};
pub use exponential::{ExponentialModel, PowerLawModel};
pub use peak::{GaussianModel, LorentzianModel, PseudoVoigtModel};
pub use polynomial::{ConstantModel, LinearModel, PolynomialModel, QuadraticModel};
pub use step::{LinearStepModel, SigmoidModel};

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
    LinearModel::new(prefix, 1, with_init)
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
    QuadraticModel::new(prefix, 2, with_init)
}

/// Create a Gaussian peak model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_baseline` - Whether to include a baseline parameter
///
/// # Returns
///
/// * A Gaussian model (y = amplitude * exp(-(x-center)^2 / (2*sigma^2)) + baseline)
pub fn gaussian_model(prefix: &str, with_baseline: bool) -> GaussianModel {
    GaussianModel::new(prefix, with_baseline)
}

/// Create a Lorentzian peak model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_baseline` - Whether to include a baseline parameter
///
/// # Returns
///
/// * A Lorentzian model (y = amplitude * gamma^2 / ((x-center)^2 + gamma^2) + baseline)
pub fn lorentzian_model(prefix: &str, with_baseline: bool) -> LorentzianModel {
    LorentzianModel::new(prefix, with_baseline)
}

/// Create a PseudoVoigt peak model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_baseline` - Whether to include a baseline parameter
///
/// # Returns
///
/// * A PseudoVoigt model (weighted sum of Gaussian and Lorentzian)
pub fn pseudovoigt_model(prefix: &str, with_baseline: bool) -> PseudoVoigtModel {
    PseudoVoigtModel::new(prefix, with_baseline)
}

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

/// Create a linear step model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_baseline` - Whether to include a baseline parameter
///
/// # Returns
///
/// * A linear step model (y = amplitude * (x-center)/abs(x-center)+sigma) + baseline)
pub fn linear_step_model(prefix: &str, with_baseline: bool) -> LinearStepModel {
    LinearStepModel::new(prefix, with_baseline)
}

/// Create a sigmoid model
///
/// # Arguments
///
/// * `prefix` - The prefix for parameter names
/// * `with_baseline` - Whether to include a baseline parameter
///
/// # Returns
///
/// * A sigmoid model (y = amplitude / (1 + exp(-(x-center)/sigma)) + baseline)
pub fn sigmoid_model(prefix: &str, with_baseline: bool) -> SigmoidModel {
    SigmoidModel::new(prefix, with_baseline)
}

// Rectangle model is not implemented yet

// Composite models - placeholder for future implementation
