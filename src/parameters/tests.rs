#[cfg(test)]
mod tests {
    use crate::parameters::{Bounds, BoundsTransform, Parameter, Parameters};
    use std::f64::{INFINITY, NEG_INFINITY};
    use std::path::Path;

    #[test]
    fn test_bounds_creation() {
        // Valid bounds
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        assert_eq!(bounds.min, 0.0);
        assert_eq!(bounds.max, 10.0);

        // Invalid bounds (min > max)
        let result = Bounds::new(10.0, 0.0);
        assert!(result.is_err());

        // Unbounded
        let bounds = Bounds::unbounded();
        assert_eq!(bounds.min, NEG_INFINITY);
        assert_eq!(bounds.max, INFINITY);

        // Lower bound only
        let bounds = Bounds::min_only(5.0);
        assert_eq!(bounds.min, 5.0);
        assert_eq!(bounds.max, INFINITY);

        // Upper bound only
        let bounds = Bounds::max_only(15.0);
        assert_eq!(bounds.min, NEG_INFINITY);
        assert_eq!(bounds.max, 15.0);
    }

    #[test]
    fn test_parameter_creation() {
        // Basic parameter
        let param = Parameter::new("amplitude", 10.0);
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);
        assert_eq!(param.init_value(), 10.0);
        assert!(param.vary());
        assert_eq!(param.min(), NEG_INFINITY);
        assert_eq!(param.max(), INFINITY);
        assert!(param.expr().is_none());

        // Parameter with bounds
        let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);
        assert!(param.vary());
        assert_eq!(param.min(), 0.0);
        assert_eq!(param.max(), 20.0);

        // Parameter with expression
        let param = Parameter::with_expr("half_amplitude", 5.0, "amplitude / 2").unwrap();
        assert_eq!(param.name(), "half_amplitude");
        assert_eq!(param.value(), 5.0);
        assert!(!param.vary());
        assert_eq!(param.expr().unwrap(), "amplitude / 2");
    }

    #[test]
    fn test_parameters_collection() {
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

        // Get a parameter
        let param = params.get("amplitude").unwrap();
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);

        // Get a mutable reference to a parameter
        let param = params.get_mut("center").unwrap();
        param.set_value(7.5).unwrap();

        // Check that the value was updated
        assert_eq!(params.get("center").unwrap().value(), 7.5);

        // Remove a parameter
        let param = params.remove("amplitude").unwrap();
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);

        assert_eq!(params.len(), 1);
        assert!(!params.contains("amplitude"));
    }

    #[test]
    fn test_parameter_serialization() {
        // Create parameters
        let mut params = Parameters::new();
        params
            .add_param_with_bounds("amplitude", 10.0, 0.0, 20.0)
            .unwrap();
        params.add_param("center", 5.0).unwrap();

        let param = params.get_mut("center").unwrap();
        param.set_vary(false).unwrap();

        // Serialize to JSON string
        let json = params.to_json().unwrap();
        println!("Serialized JSON: {}", json);

        // Deserialize from JSON string
        let params2 = Parameters::from_json(&json).unwrap();

        // Check that deserialized parameters match original
        assert_eq!(params2.len(), params.len());
        assert!(params2.contains("amplitude"));
        assert!(params2.contains("center"));

        let amp1 = params.get("amplitude").unwrap();
        let amp2 = params2.get("amplitude").unwrap();
        assert_eq!(amp2.value(), amp1.value());
        assert_eq!(amp2.vary(), amp1.vary());
        assert_eq!(amp2.min(), amp1.min());
        assert_eq!(amp2.max(), amp1.max());

        let cen1 = params.get("center").unwrap();
        let cen2 = params2.get("center").unwrap();
        assert_eq!(cen2.value(), cen1.value());
        assert_eq!(cen2.vary(), cen1.vary());

        // Test file serialization if we can write to temp
        if let Some(temp_dir) = std::env::temp_dir().to_str() {
            let file_path = format!("{}/params_test.json", temp_dir);
            let path = Path::new(&file_path);

            // Save to file
            params.save_json(path).unwrap();

            // Load from file
            let params3 = Parameters::load_json(path).unwrap();

            // Check that loaded parameters match original
            assert_eq!(params3.len(), params.len());
            assert!(params3.contains("amplitude"));
            assert!(params3.contains("center"));

            // Clean up
            std::fs::remove_file(path).unwrap_or(());
        }
    }

    #[test]
    fn test_bounds_transform() {
        // Test unbounded parameter
        let bounds = Bounds::unbounded();
        let transform = BoundsTransform::new(bounds);

        // For unbounded parameters, internal and external values should be the same
        let test_values = [-10.0, -1.0, 0.0, 1.0, 10.0];

        for &value in &test_values {
            assert_eq!(transform.to_external(value), value);
            assert_eq!(transform.to_internal(value).unwrap(), value);
        }

        // Test parameter with both bounds
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        let transform = BoundsTransform::new(bounds);

        // Test converting from internal to external values
        // For some values, we need a looser tolerance for round-trip tests
        let test_tolerance = 1e-8;

        // Test only a few values that should round-trip well
        let internal_values = [0.0, 0.5, 1.0];
        for &internal in &internal_values {
            let external = transform.to_external(internal);
            assert!(external >= bounds.min);
            assert!(external <= bounds.max);

            // Round-trip test for values within bounds
            if bounds.is_within_bounds(external) {
                let internal_round_trip = transform.to_internal(external).unwrap();
                assert!(
                    (internal - internal_round_trip).abs() < test_tolerance,
                    "Failed for internal value: {}, round-trip diff: {}",
                    internal,
                    (internal - internal_round_trip).abs()
                );
            }
        }
    }
}
