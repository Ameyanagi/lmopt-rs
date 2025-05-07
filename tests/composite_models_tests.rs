//! Comprehensive tests for composite models.
//!
//! These tests evaluate the behavior and correctness of composite models
//! created by combining different models with operations like addition and multiplication.

use approx::assert_relative_eq;
use lmopt_rs::error::Result;
use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::models::{
    add, composite_with_shared_params, gaussian_model, linear_model, lorentzian_model, multiply,
    ConstantModel, GaussianModel, LinearModel, LorentzianModel, Operation,
};
use lmopt_rs::parameters::{Parameter, Parameters};
use ndarray::{array, Array1};
use std::collections::HashMap;

#[test]
fn test_add_two_peaks() -> Result<()> {
    // Create a Gaussian and a Lorentzian peak
    let mut gaussian = GaussianModel::new("g_", true);
    let mut lorentzian = LorentzianModel::new("l_", true);

    // Set parameters for Gaussian
    gaussian
        .parameters_mut()
        .get_mut("g_amplitude")
        .unwrap()
        .set_value(3.0)?;
    gaussian
        .parameters_mut()
        .get_mut("g_center")
        .unwrap()
        .set_value(-2.0)?;
    gaussian
        .parameters_mut()
        .get_mut("g_sigma")
        .unwrap()
        .set_value(1.0)?;
    gaussian
        .parameters_mut()
        .get_mut("g_baseline")
        .unwrap()
        .set_value(0.0)?;

    // Set parameters for Lorentzian
    lorentzian
        .parameters_mut()
        .get_mut("l_amplitude")
        .unwrap()
        .set_value(2.0)?;
    lorentzian
        .parameters_mut()
        .get_mut("l_center")
        .unwrap()
        .set_value(2.0)?;
    lorentzian
        .parameters_mut()
        .get_mut("l_gamma")
        .unwrap()
        .set_value(0.5)?;
    lorentzian
        .parameters_mut()
        .get_mut("l_baseline")
        .unwrap()
        .set_value(0.0)?;

    // Create composite model: gaussian + lorentzian
    let composite = add(gaussian, lorentzian, None, None)?;

    // Test evaluation
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = composite.eval(&x)?;

    // Verify the model has all the expected parameters
    let params = composite.parameters();
    assert!(params.get("g_amplitude").is_some());
    assert!(params.get("g_center").is_some());
    assert!(params.get("g_sigma").is_some());
    assert!(params.get("g_baseline").is_some());
    assert!(params.get("l_amplitude").is_some());
    assert!(params.get("l_center").is_some());
    assert!(params.get("l_gamma").is_some());
    assert!(params.get("l_baseline").is_some());

    // Check that we have distinct peaks at the expected locations
    // Find the local maxima
    let mut maxima = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            maxima.push((x[i], y[i]));
        }
    }

    // We should have at least two peaks
    assert!(
        maxima.len() >= 2,
        "Expected at least two peaks, found {}",
        maxima.len()
    );

    // Sort maxima by x position
    maxima.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Check if the peaks are roughly at the expected positions (-2 and 2)
    let left_peak = maxima.first().unwrap();
    let right_peak = maxima.last().unwrap();

    assert_relative_eq!(left_peak.0, -2.0, epsilon = 0.5);
    assert_relative_eq!(right_peak.0, 2.0, epsilon = 0.5);

    Ok(())
}

#[test]
fn test_multiply_models() -> Result<()> {
    // Create a Gaussian and a linear model
    let mut gaussian = GaussianModel::new("g_", false); // No baseline
    let mut linear = LinearModel::new("l_", false); // No baseline

    // Set parameters for Gaussian
    gaussian
        .parameters_mut()
        .get_mut("g_amplitude")
        .unwrap()
        .set_value(1.0)?;
    gaussian
        .parameters_mut()
        .get_mut("g_center")
        .unwrap()
        .set_value(0.0)?;
    gaussian
        .parameters_mut()
        .get_mut("g_sigma")
        .unwrap()
        .set_value(1.0)?;

    // Set parameters for Linear
    linear
        .parameters_mut()
        .get_mut("l_c0")
        .unwrap()
        .set_value(0.0)?;
    linear
        .parameters_mut()
        .get_mut("l_c1")
        .unwrap()
        .set_value(0.5)?;

    // Create composite model: gaussian * linear
    let composite = multiply(gaussian, linear, None, None)?;

    // Test evaluation
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = composite.eval(&x)?;

    // Calculate expected values manually
    // Gaussian: exp(-0.5 * x^2)
    // Linear: 0.5 * x
    // Product: exp(-0.5 * x^2) * (0.5 * x)
    let expected = vec![
        (-0.5 * 4.0).exp() * (-2.0 * 0.5), // x = -2
        (-0.5 * 1.0).exp() * (-1.0 * 0.5), // x = -1
        (-0.5 * 0.0).exp() * (0.0 * 0.5),  // x = 0 (should be 0)
        (-0.5 * 1.0).exp() * (1.0 * 0.5),  // x = 1
        (-0.5 * 4.0).exp() * (2.0 * 0.5),  // x = 2
    ];

    // Check the results
    assert_eq!(y.len(), 5);
    for i in 0..5 {
        assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
    }

    // The function should be antisymmetric around 0
    assert_relative_eq!(y[0], -y[4], epsilon = 1e-10);
    assert_relative_eq!(y[1], -y[3], epsilon = 1e-10);
    assert_relative_eq!(y[2], 0.0, epsilon = 1e-10);

    Ok(())
}

#[test]
fn test_shared_parameters() -> Result<()> {
    // Create two Gaussian models with shared parameters
    let mut gaussian1 = GaussianModel::new("", true);
    let mut gaussian2 = GaussianModel::new("", true);

    // Create mapping of shared parameters
    let mut shared_params = HashMap::new();
    shared_params.insert("center".to_string(), "center".to_string());
    shared_params.insert("sigma".to_string(), "sigma".to_string());

    // Create composite model with shared parameters
    let mut composite =
        composite_with_shared_params(gaussian1, gaussian2, Operation::Add, shared_params)?;

    // Set parameters for both Gaussians
    // They should share center and sigma
    composite
        .parameters_mut()
        .get_mut("left_amplitude")
        .unwrap()
        .set_value(2.0)?;
    composite
        .parameters_mut()
        .get_mut("left_center")
        .unwrap()
        .set_value(0.0)?;
    composite
        .parameters_mut()
        .get_mut("left_sigma")
        .unwrap()
        .set_value(1.0)?;
    composite
        .parameters_mut()
        .get_mut("left_baseline")
        .unwrap()
        .set_value(0.0)?;

    composite
        .parameters_mut()
        .get_mut("right_amplitude")
        .unwrap()
        .set_value(1.0)?;
    composite
        .parameters_mut()
        .get_mut("right_baseline")
        .unwrap()
        .set_value(0.5)?;

    // Test evaluation
    let x = array![-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = composite.eval(&x)?;

    // Expected values: gaussian1(x) + gaussian2(x) with shared center and sigma
    // gaussian1: 2.0 * exp(-0.5 * x^2) + 0.0
    // gaussian2: 1.0 * exp(-0.5 * x^2) + 0.5
    let expected = vec![
        (2.0 * (-0.5 * 4.0).exp() + 0.0) + (1.0 * (-0.5 * 4.0).exp() + 0.5), // x = -2
        (2.0 * (-0.5 * 1.0).exp() + 0.0) + (1.0 * (-0.5 * 1.0).exp() + 0.5), // x = -1
        (2.0 * (-0.5 * 0.0).exp() + 0.0) + (1.0 * (-0.5 * 0.0).exp() + 0.5), // x = 0
        (2.0 * (-0.5 * 1.0).exp() + 0.0) + (1.0 * (-0.5 * 1.0).exp() + 0.5), // x = 1
        (2.0 * (-0.5 * 4.0).exp() + 0.0) + (1.0 * (-0.5 * 4.0).exp() + 0.5), // x = 2
    ];

    // Check the results
    assert_eq!(y.len(), 5);
    for i in 0..5 {
        assert_relative_eq!(y[i], expected[i], epsilon = 1e-10);
    }

    // Change a shared parameter and verify both models update
    composite
        .parameters_mut()
        .get_mut("left_center")
        .unwrap()
        .set_value(1.0)?;
    let y_shifted = composite.eval(&x)?;

    // Expected values with shifted center
    // gaussian1: 2.0 * exp(-0.5 * (x-1)^2) + 0.0
    // gaussian2: 1.0 * exp(-0.5 * (x-1)^2) + 0.5
    let expected_shifted = vec![
        (2.0 * (-0.5 * 9.0).exp() + 0.0) + (1.0 * (-0.5 * 9.0).exp() + 0.5), // x = -2
        (2.0 * (-0.5 * 4.0).exp() + 0.0) + (1.0 * (-0.5 * 4.0).exp() + 0.5), // x = -1
        (2.0 * (-0.5 * 1.0).exp() + 0.0) + (1.0 * (-0.5 * 1.0).exp() + 0.5), // x = 0
        (2.0 * (-0.5 * 0.0).exp() + 0.0) + (1.0 * (-0.5 * 0.0).exp() + 0.5), // x = 1
        (2.0 * (-0.5 * 1.0).exp() + 0.0) + (1.0 * (-0.5 * 1.0).exp() + 0.5), // x = 2
    ];

    // Check the results
    for i in 0..5 {
        assert_relative_eq!(y_shifted[i], expected_shifted[i], epsilon = 1e-10);
    }

    // The peak should now be centered at x = 1
    // Find where the function is maximum
    let max_idx = y_shifted
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(x[max_idx], 1.0, "Peak should be centered at x = 1");

    Ok(())
}

#[test]
fn test_complex_composition() -> Result<()> {
    // Create multiple models with prefixes
    let gaussian1 = gaussian_model("g1_", true);
    let gaussian2 = gaussian_model("g2_", true);
    let lorentzian = lorentzian_model("l_", true);
    let linear = linear_model("lin_", false);

    // Combine models in a complex structure:
    // ((gaussian1 + gaussian2) * linear) + lorentzian

    // Step 1: gaussian1 + gaussian2
    let peaks_sum = add(gaussian1, gaussian2, None, None)?;

    // Step 2: (gaussian1 + gaussian2) * linear
    let modulated_peaks = multiply(peaks_sum, linear, None, None)?;

    // Step 3: ((gaussian1 + gaussian2) * linear) + lorentzian
    let final_model = add(modulated_peaks, lorentzian, None, None)?;

    // Set parameters
    let params = final_model.parameters_mut();

    // Gaussian 1 parameters
    params.get_mut("g1_amplitude").unwrap().set_value(2.0)?;
    params.get_mut("g1_center").unwrap().set_value(-2.0)?;
    params.get_mut("g1_sigma").unwrap().set_value(0.8)?;
    params.get_mut("g1_baseline").unwrap().set_value(0.0)?;

    // Gaussian 2 parameters
    params.get_mut("g2_amplitude").unwrap().set_value(3.0)?;
    params.get_mut("g2_center").unwrap().set_value(0.0)?;
    params.get_mut("g2_sigma").unwrap().set_value(1.0)?;
    params.get_mut("g2_baseline").unwrap().set_value(0.0)?;

    // Linear parameters
    params.get_mut("lin_c0").unwrap().set_value(1.0)?;
    params.get_mut("lin_c1").unwrap().set_value(0.2)?;

    // Lorentzian parameters
    params.get_mut("l_amplitude").unwrap().set_value(1.5)?;
    params.get_mut("l_center").unwrap().set_value(3.0)?;
    params.get_mut("l_gamma").unwrap().set_value(0.5)?;
    params.get_mut("l_baseline").unwrap().set_value(0.1)?;

    // Evaluate the model
    let x = Array1::linspace(-5.0, 5.0, 200);
    let y = final_model.eval(&x)?;

    // The model should have peaks near -2, 0, and 3
    // Look for local maxima to verify peak locations
    let mut maxima = Vec::new();
    for i in 1..y.len() - 1 {
        if y[i] > y[i - 1] && y[i] > y[i + 1] {
            maxima.push((x[i], y[i]));
        }
    }

    // Ensure we have at least 3 peaks
    assert!(
        maxima.len() >= 3,
        "Expected at least 3 peaks, found {}",
        maxima.len()
    );

    // Sort maxima by x position
    maxima.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Check if there are peaks near the expected positions
    // These assertions are more relaxed since the peaks are modulated
    let has_peak_near = |pos: f64| maxima.iter().any(|(x, _)| (x - pos).abs() < 0.6);

    assert!(has_peak_near(-2.0), "No peak found near x = -2");
    assert!(has_peak_near(0.0), "No peak found near x = 0");
    assert!(has_peak_near(3.0), "No peak found near x = 3");

    Ok(())
}

#[test]
fn test_fitting_composite_model() -> Result<()> {
    // Create synthetic data with two Gaussian peaks
    let x = Array1::linspace(-10.0, 10.0, 200);

    // First peak: amplitude=2.0, center=-3.0, sigma=1.0
    // Second peak: amplitude=3.0, center=3.0, sigma=1.5
    // Plus baseline = 0.5
    let y: Array1<f64> = x
        .iter()
        .map(|&x_val| {
            let peak1 = 2.0 * (-(x_val + 3.0).powi(2) / (2.0 * 1.0f64.powi(2))).exp();
            let peak2 = 3.0 * (-(x_val - 3.0).powi(2) / (2.0 * 1.5f64.powi(2))).exp();
            peak1 + peak2 + 0.5
        })
        .collect();

    // Create a composite model with two Gaussian peaks
    let mut gaussian1 = gaussian_model("g1_", false); // No baseline
    let mut gaussian2 = gaussian_model("g2_", false); // No baseline
    let mut baseline = ConstantModel::new("c_", true);

    // The complete model is (gaussian1 + gaussian2 + baseline)
    let sum_peaks = add(gaussian1, gaussian2, None, None)?;
    let model = add(sum_peaks, baseline, None, None)?;

    // Initial guess - intentionally somewhat off
    let params = model.parameters_mut();
    params.get_mut("g1_amplitude").unwrap().set_value(1.0)?;
    params.get_mut("g1_center").unwrap().set_value(-2.0)?;
    params.get_mut("g1_sigma").unwrap().set_value(1.2)?;

    params.get_mut("g2_amplitude").unwrap().set_value(2.0)?;
    params.get_mut("g2_center").unwrap().set_value(2.0)?;
    params.get_mut("g2_sigma").unwrap().set_value(1.2)?;

    params.get_mut("c_c0").unwrap().set_value(0.2)?;

    // Fit the model to the data
    let fit_result = fit(&model, &x, &y)?;

    // Verify the fit converged
    assert!(
        fit_result.success,
        "Fit did not converge: {}",
        fit_result.message
    );

    // Check the fitted parameters - they should be close to the true values
    let final_params = model.parameters();

    // Output for debugging
    println!("Fitted parameters:");
    println!(
        "g1_amplitude = {}",
        final_params.get("g1_amplitude").unwrap().value()
    );
    println!(
        "g1_center = {}",
        final_params.get("g1_center").unwrap().value()
    );
    println!(
        "g1_sigma = {}",
        final_params.get("g1_sigma").unwrap().value()
    );
    println!(
        "g2_amplitude = {}",
        final_params.get("g2_amplitude").unwrap().value()
    );
    println!(
        "g2_center = {}",
        final_params.get("g2_center").unwrap().value()
    );
    println!(
        "g2_sigma = {}",
        final_params.get("g2_sigma").unwrap().value()
    );
    println!("c_c0 = {}", final_params.get("c_c0").unwrap().value());

    // Verify peak 1 parameters
    assert_relative_eq!(
        final_params.get("g1_amplitude").unwrap().value(),
        2.0,
        epsilon = 0.3
    );
    assert_relative_eq!(
        final_params.get("g1_center").unwrap().value(),
        -3.0,
        epsilon = 0.3
    );
    assert_relative_eq!(
        final_params.get("g1_sigma").unwrap().value(),
        1.0,
        epsilon = 0.3
    );

    // Verify peak 2 parameters
    assert_relative_eq!(
        final_params.get("g2_amplitude").unwrap().value(),
        3.0,
        epsilon = 0.3
    );
    assert_relative_eq!(
        final_params.get("g2_center").unwrap().value(),
        3.0,
        epsilon = 0.3
    );
    assert_relative_eq!(
        final_params.get("g2_sigma").unwrap().value(),
        1.5,
        epsilon = 0.3
    );

    // Verify baseline
    assert_relative_eq!(
        final_params.get("c_c0").unwrap().value(),
        0.5,
        epsilon = 0.2
    );

    // The fit should have a small residual
    assert!(
        fit_result.cost < 0.1,
        "Fit cost too high: {}",
        fit_result.cost
    );

    Ok(())
}

#[test]
fn test_nested_composition() -> Result<()> {
    // Build a complex model with multiple layers of composition
    // (g1 + g2) * ((g3 + l1) * lin)

    // Create individual models
    let g1 = gaussian_model("g1_", false);
    let g2 = gaussian_model("g2_", false);
    let g3 = gaussian_model("g3_", false);
    let l1 = lorentzian_model("l1_", false);
    let lin = linear_model("lin_", false);

    // First level: g1 + g2
    let sum1 = add(g1, g2, None, None)?;

    // First level: g3 + l1
    let sum2 = add(g3, l1, None, None)?;

    // Second level: (g3 + l1) * lin
    let prod1 = multiply(sum2, lin, None, None)?;

    // Third level: (g1 + g2) * ((g3 + l1) * lin)
    let final_model = multiply(sum1, prod1, None, None)?;

    // Set parameters
    let params = final_model.parameters_mut();

    // g1 parameters
    params.get_mut("g1_amplitude").unwrap().set_value(1.0)?;
    params.get_mut("g1_center").unwrap().set_value(-2.0)?;
    params.get_mut("g1_sigma").unwrap().set_value(0.5)?;

    // g2 parameters
    params.get_mut("g2_amplitude").unwrap().set_value(1.0)?;
    params.get_mut("g2_center").unwrap().set_value(2.0)?;
    params.get_mut("g2_sigma").unwrap().set_value(0.5)?;

    // g3 parameters
    params.get_mut("g3_amplitude").unwrap().set_value(1.0)?;
    params.get_mut("g3_center").unwrap().set_value(0.0)?;
    params.get_mut("g3_sigma").unwrap().set_value(2.0)?;

    // l1 parameters
    params.get_mut("l1_amplitude").unwrap().set_value(0.5)?;
    params.get_mut("l1_center").unwrap().set_value(0.0)?;
    params.get_mut("l1_gamma").unwrap().set_value(1.0)?;

    // lin parameters
    params.get_mut("lin_c0").unwrap().set_value(1.0)?;
    params.get_mut("lin_c1").unwrap().set_value(0.1)?;

    // Evaluate the model
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = final_model.eval(&x)?;

    // Verify the model has the correct structure by evaluating at specific points

    // At x = -2, g1 is maximal (1.0), g2 is small,
    // g3 is moderate, l1 is moderate, lin is 0.8
    let y_minus2 = y[x.iter().position(|&val| (val + 2.0).abs() < 0.1).unwrap()];

    // At x = 2, g1 is small, g2 is maximal (1.0),
    // g3 is moderate, l1 is moderate, lin is 1.2
    let y_plus2 = y[x.iter().position(|&val| (val - 2.0).abs() < 0.1).unwrap()];

    // At x = 0, g1 is moderate, g2 is moderate,
    // g3 is maximal (1.0), l1 is maximal (0.5), lin is 1.0
    let y_0 = y[x.iter().position(|&val| val.abs() < 0.1).unwrap()];

    // Relative magnitudes should follow from the model structure
    assert!(y_minus2 > 0.0);
    assert!(y_plus2 > 0.0);
    assert!(y_0 > 0.0);

    // Count the total number of parameters in the model
    let param_count = final_model.parameters().len();
    assert_eq!(param_count, 11); // 5 models with 2-3 parameters each, minus shared names

    // Ensure all parameter prefixes are preserved
    assert!(params.get("g1_amplitude").is_some());
    assert!(params.get("g2_amplitude").is_some());
    assert!(params.get("g3_amplitude").is_some());
    assert!(params.get("l1_amplitude").is_some());
    assert!(params.get("lin_c0").is_some());

    Ok(())
}
