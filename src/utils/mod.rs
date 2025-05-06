//! Utility functions and helpers for the lmopt-rs library.

pub mod matrix_convert;
pub mod finite_difference;
pub mod autodiff;

// Re-export commonly used utilities
pub use matrix_convert::{
    ndarray_to_faer, faer_to_ndarray,
    ndarray_vec_to_faer, faer_vec_to_ndarray,
    nalgebra_to_faer, faer_to_nalgebra,
    nalgebra_to_ndarray, ndarray_to_nalgebra,
    nalgebra_vec_to_faer, faer_vec_to_nalgebra,
    nalgebra_vec_to_ndarray, ndarray_vec_to_nalgebra,
};

// Prefer autodiff implementations when available, with finite_difference as fallback
pub use autodiff::{
    jacobian, gradient, hessian,
};