//! Integration tests for the lmopt-rs library
//!
//! This module organizes all integration tests that test the library as a whole,
//! rather than individual components.

// NIST Statistical Reference Datasets tests
pub mod nist_strd;

// Comparison with lmfit-py results
pub mod lmfit_comparison;

// Real-world optimization problems
pub mod real_world;