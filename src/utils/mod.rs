//! Utility functions and helpers for the lmopt-rs library.

pub mod autodiff;
pub mod finite_difference;
pub mod matrix_convert;
pub mod parallel;

// Re-export commonly used utilities
pub use matrix_convert::{
    faer_to_nalgebra, faer_to_ndarray, faer_vec_to_nalgebra, faer_vec_to_ndarray, nalgebra_to_faer,
    nalgebra_to_ndarray, nalgebra_vec_to_faer, nalgebra_vec_to_ndarray, ndarray_to_faer,
    ndarray_to_nalgebra, ndarray_vec_to_faer, ndarray_vec_to_nalgebra,
};

// Prefer autodiff implementations when available, with finite_difference as fallback
pub use autodiff::{gradient, hessian, jacobian};

// Parallel implementations
pub use parallel::{
    eval_residuals_parallel, gradient_parallel, hessian_parallel, jacobian_central_parallel,
    jacobian_parallel,
};
