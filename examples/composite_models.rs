//! Example of using composite models for advanced fitting.
//!
//! This example demonstrates how to combine multiple models to create
//! more complex functions, such as multi-peak spectra, background-corrected
//! data, and more.

use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::models::{
    add, composite_with_shared_params, multiply, ExponentialModel, LinearModel, Operation,
};
use lmopt_rs::parameters::Parameters;

use ndarray::Array1;
use rand::Rng;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Composite models example");
    println!("=======================\n");

    // Create test data for our examples
    let mut rng = rand::thread_rng();

    // 1. Multiple peaks with a linear background
    println!("1. Multiple peaks with a linear background");
    println!("----------------------------------------");

    // Create x-array for testing
    let x_multi = Array1::linspace(0.0, 10.0, 100);
    let mut y_multi = Array1::zeros(100);

    // Generate data: two Gaussians + linear background
    // First Gaussian: amplitude=3.0, center=3.0, sigma=0.7
    // Second Gaussian: amplitude=2.0, center=7.0, sigma=1.0
    // Linear background: y = 0.5 + 0.2 * x
    for i in 0..100 {
        let x: f64 = x_multi[i];

        // First Gaussian
        let g1 = 3.0 * (-((x - 3.0) as f64).powi(2) / (2.0 * 0.7_f64.powi(2))).exp();

        // Second Gaussian
        let g2 = 2.0 * (-((x - 7.0) as f64).powi(2) / (2.0 * 1.0_f64.powi(2))).exp();

        // Linear background
        let bg = 0.5 + 0.2 * x;

        // Combine with noise
        let noise = rng.gen_range(-0.2..0.2);
        y_multi[i] = g1 + g2 + bg + noise;
    }

    // Create custom Gaussian model 1
    let mut params1 = Parameters::new();
    params1.add_param("g1_amplitude", 2.0)?;
    params1.add_param("g1_center", 3.0)?;
    params1.add_param("g1_sigma", 1.0)?;
    params1.add_param("g1_baseline", 0.0)?;

    let gaussian1 = BaseModel::new(params1, |params, x| {
        let amplitude = params.get("g1_amplitude").unwrap().value();
        let center = params.get("g1_center").unwrap().value();
        let sigma = params.get("g1_sigma").unwrap().value();
        let baseline = params.get("g1_baseline").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    // Create custom Gaussian model 2
    let mut params2 = Parameters::new();
    params2.add_param("g2_amplitude", 1.5)?;
    params2.add_param("g2_center", 7.0)?;
    params2.add_param("g2_sigma", 1.2)?;
    params2.add_param("g2_baseline", 0.0)?;

    let gaussian2 = BaseModel::new(params2, |params, x| {
        let amplitude = params.get("g2_amplitude").unwrap().value();
        let center = params.get("g2_center").unwrap().value();
        let sigma = params.get("g2_sigma").unwrap().value();
        let baseline = params.get("g2_baseline").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    // Create linear background model
    let mut linear_bg = LinearModel::new("bg_", true);
    linear_bg
        .parameters_mut()
        .get_mut("bg_c0")
        .unwrap()
        .set_value(0.0)
        .unwrap();
    linear_bg
        .parameters_mut()
        .get_mut("bg_c1")
        .unwrap()
        .set_value(0.1)
        .unwrap();

    // First combine the two Gaussians
    let two_gaussians = add(gaussian1, gaussian2, None, None)?;

    // Then add the linear background
    let mut model = add(two_gaussians, linear_bg, None, None)?;

    // Fit the model to the data
    let result = fit(&mut model, x_multi.clone(), y_multi.clone())?;

    // Output the results
    println!("Fit success: {}", result.success);
    println!("Final sum of squared residuals: {:.5}", result.cost);
    println!("\nFitted parameters:");

    // First Gaussian parameters
    println!("  First Gaussian:");
    println!(
        "    Amplitude: {:.3}",
        model.parameters().get("g1_amplitude").unwrap().value()
    );
    println!(
        "    Center: {:.3}",
        model.parameters().get("g1_center").unwrap().value()
    );
    println!(
        "    Sigma: {:.3}",
        model.parameters().get("g1_sigma").unwrap().value()
    );

    // Second Gaussian parameters
    println!("  Second Gaussian:");
    println!(
        "    Amplitude: {:.3}",
        model.parameters().get("g2_amplitude").unwrap().value()
    );
    println!(
        "    Center: {:.3}",
        model.parameters().get("g2_center").unwrap().value()
    );
    println!(
        "    Sigma: {:.3}",
        model.parameters().get("g2_sigma").unwrap().value()
    );

    // Background parameters
    println!("  Linear Background:");
    println!(
        "    Intercept: {:.3}",
        model.parameters().get("bg_c0").unwrap().value()
    );
    println!(
        "    Slope: {:.3}",
        model.parameters().get("bg_c1").unwrap().value()
    );

    // Compare to true values
    println!("\nComparison with true values:");
    println!("  First Gaussian: amplitude=3.0, center=3.0, sigma=0.7");
    println!("  Second Gaussian: amplitude=2.0, center=7.0, sigma=1.0");
    println!("  Linear background: intercept=0.5, slope=0.2");
    println!();

    // 2. Product model: Exponential decay modulated by sine wave
    println!("2. Product model: Exponential decay modulated by sine wave");
    println!("-----------------------------------------------------");

    // Create data for a product model
    let x_prod = Array1::linspace(0.0, 20.0, 100);
    let mut y_prod = Array1::zeros(100);

    // Generate data: exponential decay * sinusoid
    // y = 5.0 * exp(-x/5.0) * (1.0 + 0.5 * sin(x))
    for i in 0..100 {
        let x: f64 = x_prod[i];
        let exp_decay = 5.0 * (-x / 5.0).exp();
        let sinusoid = 1.0 + 0.5 * (x).sin();
        let noise = rng.gen_range(-0.1..0.1);
        y_prod[i] = exp_decay * sinusoid + noise;
    }

    // Create a custom sine wave model
    let mut sine_params = Parameters::new();
    sine_params.add_param("amplitude", 0.3)?;
    sine_params.add_param("frequency", 1.0)?;
    sine_params.add_param("phase", 0.0)?;
    sine_params.add_param("offset", 1.0)?;

    let sine_model = BaseModel::new(sine_params, |params, x| {
        let amplitude = params.get("amplitude").unwrap().value();
        let frequency = params.get("frequency").unwrap().value();
        let phase = params.get("phase").unwrap().value();
        let offset = params.get("offset").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| offset + amplitude * (frequency * x_val + phase).sin())
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    // Create an exponential decay model
    let mut exp_model = ExponentialModel::new("exp_", true);
    exp_model
        .parameters_mut()
        .get_mut("exp_amplitude")
        .unwrap()
        .set_value(5.0)
        .unwrap();
    exp_model
        .parameters_mut()
        .get_mut("exp_decay")
        .unwrap()
        .set_value(5.0)
        .unwrap();
    exp_model
        .parameters_mut()
        .get_mut("exp_baseline")
        .unwrap()
        .set_value(0.0)
        .unwrap();

    // Combine the models with multiplication
    let mut product_model = multiply(exp_model, sine_model, None, None)?;

    // Fit the model to the data
    let product_result = fit(&mut product_model, x_prod.clone(), y_prod.clone())?;

    // Output the results
    println!("Fit success: {}", product_result.success);
    println!("Final sum of squared residuals: {:.5}", product_result.cost);
    println!("\nFitted parameters:");

    // Exponential parameters
    println!("  Exponential:");
    println!(
        "    Amplitude: {:.3}",
        product_model
            .parameters()
            .get("exp_amplitude")
            .unwrap()
            .value()
    );
    println!(
        "    Decay: {:.3}",
        product_model.parameters().get("exp_decay").unwrap().value()
    );

    // Sine wave parameters
    println!("  Sine wave:");
    println!(
        "    Amplitude: {:.3}",
        product_model.parameters().get("amplitude").unwrap().value()
    );
    println!(
        "    Frequency: {:.3}",
        product_model.parameters().get("frequency").unwrap().value()
    );
    println!(
        "    Phase: {:.3}",
        product_model.parameters().get("phase").unwrap().value()
    );
    println!(
        "    Offset: {:.3}",
        product_model.parameters().get("offset").unwrap().value()
    );

    // Compare to true values
    println!("\nComparison with true values:");
    println!("  Exponential: amplitude=5.0, decay=5.0");
    println!("  Sine wave: amplitude=0.5, frequency=1.0, phase=0.0, offset=1.0");
    println!();

    // 3. Model with shared parameters
    println!("3. Model with shared parameters");
    println!("----------------------------");

    // Create data for two peaks with shared width
    let x_shared = Array1::linspace(0.0, 10.0, 100);
    let mut y_shared = Array1::zeros(100);

    // Generate data: two Gaussians with the same width
    // First Gaussian: amplitude=2.5, center=3.0, sigma=0.8
    // Second Gaussian: amplitude=1.5, center=7.0, sigma=0.8 (same sigma)
    for i in 0..100 {
        let x: f64 = x_shared[i];

        // First Gaussian
        let g1 = 2.5 * (-((x - 3.0) as f64).powi(2) / (2.0 * 0.8_f64.powi(2))).exp();

        // Second Gaussian (same sigma)
        let g2 = 1.5 * (-((x - 7.0) as f64).powi(2) / (2.0 * 0.8_f64.powi(2))).exp();

        // Combine with noise
        let noise = rng.gen_range(-0.1..0.1);
        y_shared[i] = g1 + g2 + noise;
    }

    // Create two custom Gaussian models
    let mut params_g1 = Parameters::new();
    params_g1.add_param("left_amplitude", 2.0)?;
    params_g1.add_param("left_center", 3.0)?;
    params_g1.add_param("left_sigma", 1.0)?;
    params_g1.add_param("left_baseline", 0.0)?;

    let g1 = BaseModel::new(params_g1, |params, x| {
        let amplitude = params.get("left_amplitude").unwrap().value();
        let center = params.get("left_center").unwrap().value();
        let sigma = params.get("left_sigma").unwrap().value();
        let baseline = params.get("left_baseline").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    let mut params_g2 = Parameters::new();
    params_g2.add_param("right_amplitude", 1.0)?;
    params_g2.add_param("right_center", 7.0)?;
    params_g2.add_param("right_sigma", 1.0)?; // Will be shared with left_sigma
    params_g2.add_param("right_baseline", 0.0)?;

    let g2 = BaseModel::new(params_g2, |params, x| {
        let amplitude = params.get("right_amplitude").unwrap().value();
        let center = params.get("right_center").unwrap().value();
        let sigma = params.get("right_sigma").unwrap().value();
        let baseline = params.get("right_baseline").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    // Create a map of shared parameters
    let mut shared_params = HashMap::new();
    shared_params.insert("left_sigma".to_string(), "right_sigma".to_string());

    // Create a composite model with shared sigma
    let mut shared_model = composite_with_shared_params(g1, g2, Operation::Add, shared_params)?;

    // Fit the model to the data
    let shared_result = fit(&mut shared_model, x_shared.clone(), y_shared.clone())?;

    // Output the results
    println!("Fit success: {}", shared_result.success);
    println!("Final sum of squared residuals: {:.5}", shared_result.cost);
    println!("\nFitted parameters:");

    // Check if parameters exist and display them
    if let Some(param) = shared_model.parameters().get("left_amplitude") {
        println!("  First Gaussian:");
        println!("    Amplitude: {:.3}", param.value());

        if let Some(center) = shared_model.parameters().get("left_center") {
            println!("    Center: {:.3}", center.value());
        }
    }

    if let Some(param) = shared_model.parameters().get("right_amplitude") {
        println!("  Second Gaussian:");
        println!("    Amplitude: {:.3}", param.value());

        if let Some(center) = shared_model.parameters().get("right_center") {
            println!("    Center: {:.3}", center.value());
        }
    }

    // Shared sigma parameter
    if let Some(sigma) = shared_model.parameters().get("left_sigma") {
        println!("  Shared parameter:");
        println!("    Sigma: {:.3}", sigma.value());
    }

    // Compare to true values
    println!("\nComparison with true values:");
    println!("  First Gaussian: amplitude=2.5, center=3.0, sigma=0.8");
    println!("  Second Gaussian: amplitude=1.5, center=7.0, sigma=0.8");

    println!("\nExample completed successfully!");
    Ok(())
}
