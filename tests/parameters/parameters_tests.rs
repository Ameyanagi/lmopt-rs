//! Integration tests for the Parameters collection
//!
//! These tests verify that the Parameters collection behaves correctly in various scenarios.

use lmopt_rs::parameters::{Parameter, Parameters};
use std::f64::{INFINITY, NEG_INFINITY};

#[test]
fn test_parameters_basic_operations() {
    // Create an empty parameters collection
    let mut params = Parameters::new();
    assert_eq!(params.len(), 0);
    assert!(params.is_empty());

    // Add a parameter
    let param = Parameter::new("amplitude", 10.0);
    params.add(param).unwrap();

    assert_eq!(params.len(), 1);
    assert!(!params.is_empty());
    assert!(params.contains("amplitude"));

    // Add another parameter
    params.add_param("center", 5.0).unwrap();

    assert_eq!(params.len(), 2);
    assert!(params.contains("center"));

    // Add a parameter with bounds
    params
        .add_param_with_bounds("sigma", 2.0, 0.1, 10.0)
        .unwrap();

    assert_eq!(params.len(), 3);
    assert!(params.contains("sigma"));

    // Get a parameter
    let param = params.get("amplitude").unwrap();
    assert_eq!(param.name(), "amplitude");
    assert_eq!(param.value(), 10.0);

    // Get a parameter that doesn't exist
    assert!(params.get("nonexistent").is_none());

    // Get a mutable reference to a parameter
    let param = params.get_mut("center").unwrap();
    param.set_value(7.5).unwrap();

    // Check that the value was updated
    assert_eq!(params.get("center").unwrap().value(), 7.5);

    // Remove a parameter
    let param = params.remove("amplitude").unwrap();
    assert_eq!(param.name(), "amplitude");
    assert_eq!(param.value(), 10.0);

    assert_eq!(params.len(), 2);
    assert!(!params.contains("amplitude"));

    // Get parameter names and values
    let names = params.names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"center".to_string()));
    assert!(names.contains(&"sigma".to_string()));

    let values = params.values();
    assert_eq!(values.len(), 2);
    assert!(values.contains(&7.5));
    assert!(values.contains(&2.0));
}

#[test]
fn test_parameters_varying_fixed() {
    // Create a parameters collection
    let mut params = Parameters::new();
    params.add_param("amplitude", 10.0).unwrap();
    params.add_param("center", 5.0).unwrap();
    params.add_param("sigma", 2.0).unwrap();

    // Initially, all parameters vary
    let varying = params.varying();
    assert_eq!(varying.len(), 3);
    assert_eq!(params.fixed().len(), 0);

    // Fix one parameter
    params.get_mut("center").unwrap().set_vary(false).unwrap();

    // Now, two parameters vary and one is fixed
    let varying = params.varying();
    assert_eq!(varying.len(), 2);
    assert!(varying.iter().any(|p| p.name() == "amplitude"));
    assert!(varying.iter().any(|p| p.name() == "sigma"));

    let fixed = params.fixed();
    assert_eq!(fixed.len(), 1);
    assert_eq!(fixed[0].name(), "center");

    // Fix another parameter
    params.get_mut("sigma").unwrap().set_vary(false).unwrap();

    // Now, one parameter varies and two are fixed
    let varying = params.varying();
    assert_eq!(varying.len(), 1);
    assert_eq!(varying[0].name(), "amplitude");

    let fixed = params.fixed();
    assert_eq!(fixed.len(), 2);
    assert!(fixed.iter().any(|p| p.name() == "center"));
    assert!(fixed.iter().any(|p| p.name() == "sigma"));
}

#[test]
fn test_parameters_varying_values() {
    // Create a parameters collection
    let mut params = Parameters::new();
    params.add_param("amplitude", 10.0).unwrap();
    params.add_param("center", 5.0).unwrap();

    // Fix one parameter
    params.get_mut("center").unwrap().set_vary(false).unwrap();

    // Get varying values
    let varying_values = params.varying_values();
    assert_eq!(varying_values.len(), 1);
    assert_eq!(varying_values[0].0, "amplitude");
    assert_eq!(varying_values[0].1, 10.0);

    // Get internal values
    let internal_values = params.varying_internal_values().unwrap();
    assert_eq!(internal_values.len(), 1);
    assert_eq!(internal_values[0].0, "amplitude");

    // For unbounded parameters, internal and external values should be the same
    assert_eq!(internal_values[0].1, 10.0);

    // Update from internal values
    let values = vec![15.0];
    params.update_from_internal(&values).unwrap();

    // Check that the value was updated
    assert_eq!(params.get("amplitude").unwrap().value(), 15.0);
    assert_eq!(params.get("center").unwrap().value(), 5.0);
}

#[test]
fn test_parameters_reset() {
    // Create a parameters collection
    let mut params = Parameters::new();
    params.add_param("amplitude", 10.0).unwrap();
    params.add_param("center", 5.0).unwrap();

    // Change parameter values
    params
        .get_mut("amplitude")
        .unwrap()
        .set_value(15.0)
        .unwrap();
    params.get_mut("center").unwrap().set_value(7.5).unwrap();

    // Reset parameters
    params.reset();

    // Check that values were reset
    assert_eq!(params.get("amplitude").unwrap().value(), 10.0);
    assert_eq!(params.get("center").unwrap().value(), 5.0);
}

#[test]
fn test_parameters_bounds_handling() {
    // Create a parameters collection with bounded parameters
    let mut params = Parameters::new();
    params
        .add_param_with_bounds("amplitude", 10.0, 0.0, 20.0)
        .unwrap();
    params
        .add_param_with_bounds("center", 5.0, 0.0, 10.0)
        .unwrap();

    // Get internal values
    let internal_values = params.varying_internal_values().unwrap();
    assert_eq!(internal_values.len(), 2);

    // Internal values should be different from external values due to bounds transform
    assert_ne!(internal_values[0].1, 10.0);
    assert_ne!(internal_values[1].1, 5.0);

    // Round-trip should preserve values
    let values = vec![internal_values[0].1, internal_values[1].1];
    params.update_from_internal(&values).unwrap();

    // Values should be unchanged
    assert!((params.get("amplitude").unwrap().value() - 10.0).abs() < 1e-10);
    assert!((params.get("center").unwrap().value() - 5.0).abs() < 1e-10);

    // Test various internal values to ensure they're mapped within bounds
    let test_values = vec![-10.0, -1.0, 0.0, 1.0, 10.0];

    for &internal in &test_values {
        let values = vec![internal, internal];
        params.update_from_internal(&values).unwrap();

        // Values should be within bounds
        let amplitude = params.get("amplitude").unwrap().value();
        let center = params.get("center").unwrap().value();

        assert!(amplitude >= 0.0 && amplitude <= 20.0);
        assert!(center >= 0.0 && center <= 10.0);
    }
}
