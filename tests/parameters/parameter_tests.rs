//! Integration tests for the Parameter struct
//!
//! These tests verify that the Parameter struct behaves correctly in various scenarios.

use lmopt_rs::parameters::{Bounds, BoundsTransform, Parameter};
use std::f64::{INFINITY, NEG_INFINITY};

#[test]
fn test_parameter_lifecycle() {
    // Create a parameter
    let mut param = Parameter::new("amplitude", 10.0);

    // Check initial state
    assert_eq!(param.name(), "amplitude");
    assert_eq!(param.value(), 10.0);
    assert!(param.vary());
    assert_eq!(param.min(), NEG_INFINITY);
    assert_eq!(param.max(), INFINITY);
    assert!(param.expr().is_none());
    assert!(param.stderr().is_none());
    assert!(param.brute_step().is_none());
    assert!(param.user_data().is_none());

    // Change value
    param.set_value(15.0).unwrap();
    assert_eq!(param.value(), 15.0);
    assert_eq!(param.init_value(), 10.0);

    // Reset to initial value
    param.reset();
    assert_eq!(param.value(), 10.0);

    // Set bounds
    param.set_bounds(0.0, 20.0).unwrap();
    assert_eq!(param.min(), 0.0);
    assert_eq!(param.max(), 20.0);

    // Set value outside bounds (should fail)
    assert!(param.set_value(-5.0).is_err());
    assert!(param.set_value(25.0).is_err());

    // Set value inside bounds (should succeed)
    assert!(param.set_value(5.0).is_ok());
    assert_eq!(param.value(), 5.0);

    // Make the parameter fixed
    param.set_vary(false).unwrap();
    assert!(!param.vary());

    // Make the parameter vary again
    param.set_vary(true).unwrap();
    assert!(param.vary());

    // Set an expression
    param.set_expr(Some("other_param * 2")).unwrap();
    assert_eq!(param.expr().unwrap(), "other_param * 2");
    assert!(!param.vary());

    // Remove the expression
    param.set_expr(None).unwrap();
    assert!(param.expr().is_none());
    assert!(!param.vary()); // Removing expression doesn't automatically make it vary

    // Set other attributes
    param.set_stderr(Some(0.5));
    assert_eq!(param.stderr().unwrap(), 0.5);

    param.set_brute_step(Some(0.1));
    assert_eq!(param.brute_step().unwrap(), 0.1);

    param.set_user_data(Some("Custom data"));
    assert_eq!(param.user_data().unwrap(), "Custom data");

    // Remove attributes
    param.set_stderr(None);
    assert!(param.stderr().is_none());

    param.set_brute_step(None);
    assert!(param.brute_step().is_none());

    param.set_user_data(None);
    assert!(param.user_data().is_none());
}

#[test]
fn test_parameter_with_bounds() {
    // Create a parameter with bounds
    let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();

    // Check that bounds were set
    assert_eq!(param.min(), 0.0);
    assert_eq!(param.max(), 20.0);

    // Invalid bounds should return an error
    assert!(Parameter::with_bounds("amplitude", 10.0, 20.0, 0.0).is_err());

    // Value outside bounds should be clamped to bounds
    let param = Parameter::with_bounds("amplitude", 30.0, 0.0, 20.0).unwrap();
    assert_eq!(param.value(), 20.0);

    let param = Parameter::with_bounds("amplitude", -10.0, 0.0, 20.0).unwrap();
    assert_eq!(param.value(), 0.0);
}

#[test]
fn test_parameter_with_expr() {
    // Create a parameter with an expression
    let param = Parameter::with_expr("half_amplitude", 5.0, "amplitude / 2").unwrap();

    // Check that the expression was set
    assert_eq!(param.expr().unwrap(), "amplitude / 2");
    assert!(!param.vary());
}

#[test]
fn test_parameter_bounds_transform() {
    // Test unbounded parameter
    let param = Parameter::new("amplitude", 10.0);

    // For unbounded parameters, internal and external values should be the same
    assert_eq!(param.to_internal().unwrap(), 10.0);
    assert_eq!(param.from_internal(15.0), 15.0);

    // Test parameter with lower bound
    let mut param = Parameter::new("amplitude", 10.0);
    param.set_bounds(5.0, INFINITY).unwrap();

    // Internal value should be transformed
    let internal = param.to_internal().unwrap();
    assert_ne!(internal, 10.0);

    // But round-trip should preserve the value
    let external = param.from_internal(internal);
    assert!((external - 10.0).abs() < 1e-10);

    // Test parameter with upper bound
    let mut param = Parameter::new("amplitude", 10.0);
    param.set_bounds(NEG_INFINITY, 15.0).unwrap();

    // Internal value should be transformed
    let internal = param.to_internal().unwrap();
    assert_ne!(internal, 10.0);

    // But round-trip should preserve the value
    let external = param.from_internal(internal);
    assert!((external - 10.0).abs() < 1e-10);

    // Test parameter with both bounds
    let mut param = Parameter::new("amplitude", 10.0);
    param.set_bounds(0.0, 20.0).unwrap();

    // Internal value should be transformed
    let internal = param.to_internal().unwrap();
    assert_ne!(internal, 10.0);

    // But round-trip should preserve the value
    let external = param.from_internal(internal);
    assert!((external - 10.0).abs() < 1e-10);

    // Test many values to ensure transformation is reversible
    let bounds = Bounds::new(0.0, 20.0).unwrap();
    let transform = BoundsTransform::new(bounds);

    let test_values = vec![0.0, 1.0, 5.0, 10.0, 15.0, 19.0, 20.0];

    for &value in &test_values {
        let internal = transform.to_internal(value).unwrap();
        let external = transform.to_external(internal);
        assert!((external - value).abs() < 1e-10);
    }
}
