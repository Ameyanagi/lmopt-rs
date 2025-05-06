//! Example of using the model system for fitting data.
//!
//! This example demonstrates how to create and fit various models using the model system.

use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::models::{ExponentialModel, QuadraticModel, SigmoidModel};
use lmopt_rs::parameters::Parameters;

use ndarray::Array1;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Fitting models example");
    println!("======================\n");

    // Create some test data points for each model type
    let mut rng = rand::thread_rng();

    // 1. Gaussian peak example (using custom model since GaussianModel isn't implemented yet)
    println!("1. Gaussian peak example (using custom model)");
    println!("-------------------------------------------");
    let x_gaussian = Array1::linspace(-5.0, 5.0, 100);
    let mut y_gaussian = Array1::zeros(100);

    // y = 3.0 * exp(-(x-1.5)^2 / (2*0.75^2)) + 0.5 + noise
    for i in 0..100 {
        let x: f64 = x_gaussian[i];
        let noise = rng.gen_range(-0.1..0.1);
        let arg = (x - 1.5) / 0.75;
        y_gaussian[i] = 3.0 * (-0.5_f64 * arg * arg).exp() + 0.5 + noise;
    }

    // Create a custom Gaussian model
    let mut params = Parameters::new();
    params.add_param("amplitude", 1.0)?;
    params.add_param("center", 0.0)?;
    params.add_param("sigma", 1.0)?;
    params.add_param("baseline", 0.0)?;

    let gaussian_model = BaseModel::new(params, |params, x| {
        let amplitude = params.get("amplitude").unwrap().value();
        let center = params.get("center").unwrap().value();
        let sigma = params.get("sigma").unwrap().value();
        let baseline = params.get("baseline").unwrap().value();

        let result = x
            .iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp() + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    let mut gaussian_model = gaussian_model;
    let gaussian_result = fit(&mut gaussian_model, x_gaussian.clone(), y_gaussian.clone())?;

    // Output results
    println!("Fit success: {}", gaussian_result.success);
    println!("Parameters:");
    println!(
        "  amplitude = {:.3}",
        gaussian_model
            .parameters()
            .get("amplitude")
            .unwrap()
            .value()
    );
    println!(
        "  center = {:.3}",
        gaussian_model.parameters().get("center").unwrap().value()
    );
    println!(
        "  sigma = {:.3}",
        gaussian_model.parameters().get("sigma").unwrap().value()
    );
    println!(
        "  baseline = {:.3}",
        gaussian_model.parameters().get("baseline").unwrap().value()
    );
    println!("Cost (residual sum): {:.6}", gaussian_result.cost);
    println!("Standard errors:");
    for name in ["amplitude", "center", "sigma", "baseline"].iter() {
        println!(
            "  {} = {:.3}",
            name,
            gaussian_result
                .standard_errors
                .get(&name.to_string())
                .unwrap_or(&0.0)
        );
    }
    println!();

    // 2. Exponential decay example
    println!("2. Exponential decay example");
    println!("---------------------------");
    let x_exp = Array1::linspace(0.0, 10.0, 50);
    let mut y_exp = Array1::zeros(50);

    // y = 5.0 * exp(-x/2.5) + 1.0 + noise
    for i in 0..50 {
        let x: f64 = x_exp[i];
        let noise = rng.gen_range(-0.2..0.2);
        y_exp[i] = 5.0 * (-x / 2.5).exp() + 1.0 + noise;
    }

    // Create and fit the exponential model
    let mut exp_model = ExponentialModel::new("", true);
    let exp_result = fit(&mut exp_model, x_exp.clone(), y_exp.clone())?;

    // Output results
    println!("Fit success: {}", exp_result.success);
    println!("Parameters:");
    println!(
        "  amplitude = {:.3}",
        exp_model.parameters().get("amplitude").unwrap().value()
    );
    println!(
        "  decay = {:.3}",
        exp_model.parameters().get("decay").unwrap().value()
    );
    println!(
        "  baseline = {:.3}",
        exp_model.parameters().get("baseline").unwrap().value()
    );
    println!("Cost (residual sum): {:.6}", exp_result.cost);
    println!("Standard errors:");
    for name in ["amplitude", "decay", "baseline"].iter() {
        println!(
            "  {} = {:.3}",
            name,
            exp_result
                .standard_errors
                .get(&name.to_string())
                .unwrap_or(&0.0)
        );
    }
    println!();

    // 3. Polynomial (quadratic) example
    println!("3. Quadratic model example");
    println!("--------------------------");
    let x_quad = Array1::linspace(-3.0, 3.0, 30);
    let mut y_quad = Array1::zeros(30);

    // y = 1.5*x^2 - 2.0*x + 0.5 + noise
    for i in 0..30 {
        let x: f64 = x_quad[i];
        let noise = rng.gen_range(-0.3..0.3);
        y_quad[i] = 1.5 * x.powi(2) - 2.0 * x + 0.5 + noise;
    }

    // Create and fit the quadratic model
    let mut quad_model = QuadraticModel::new("", true);
    let quad_result = fit(&mut quad_model, x_quad.clone(), y_quad.clone())?;

    // Output results
    println!("Fit success: {}", quad_result.success);
    println!("Parameters:");
    println!(
        "  c2 (a) = {:.3}",
        quad_model.parameters().get("c2").unwrap().value()
    );
    println!(
        "  c1 (b) = {:.3}",
        quad_model.parameters().get("c1").unwrap().value()
    );
    println!(
        "  c0 (c) = {:.3}",
        quad_model.parameters().get("c0").unwrap().value()
    );
    println!("Cost (residual sum): {:.6}", quad_result.cost);
    println!("Standard errors:");
    for name in ["c2", "c1", "c0"].iter() {
        println!(
            "  {} = {:.3}",
            name,
            quad_result
                .standard_errors
                .get(&name.to_string())
                .unwrap_or(&0.0)
        );
    }
    println!();

    // 4. Sigmoid example
    println!("4. Sigmoid model example");
    println!("------------------------");
    let x_sigmoid = Array1::linspace(-5.0, 5.0, 50);
    let mut y_sigmoid = Array1::zeros(50);

    // y = 3.0 / (1.0 + exp(-(x-1.0)/0.75)) + 0.5 + noise
    for i in 0..50 {
        let x: f64 = x_sigmoid[i];
        let noise = rng.gen_range(-0.15..0.15);
        y_sigmoid[i] = 3.0 / (1.0 + (-(x - 1.0) / 0.75).exp()) + 0.5 + noise;
    }

    // Create and fit the sigmoid model
    let mut sigmoid_model = SigmoidModel::new("", true);
    let sigmoid_result = fit(&mut sigmoid_model, x_sigmoid.clone(), y_sigmoid.clone())?;

    // Output results
    println!("Fit success: {}", sigmoid_result.success);
    println!("Parameters:");
    println!(
        "  amplitude = {:.3}",
        sigmoid_model.parameters().get("amplitude").unwrap().value()
    );
    println!(
        "  center = {:.3}",
        sigmoid_model.parameters().get("center").unwrap().value()
    );
    println!(
        "  sigma = {:.3}",
        sigmoid_model.parameters().get("sigma").unwrap().value()
    );
    println!(
        "  baseline = {:.3}",
        sigmoid_model.parameters().get("baseline").unwrap().value()
    );
    println!("Cost (residual sum): {:.6}", sigmoid_result.cost);
    println!("Standard errors:");
    for name in ["amplitude", "center", "sigma", "baseline"].iter() {
        println!(
            "  {} = {:.3}",
            name,
            sigmoid_result
                .standard_errors
                .get(&name.to_string())
                .unwrap_or(&0.0)
        );
    }
    println!();

    // 5. Multiple peaks example - fitting multiple Gaussians
    println!("5. Multiple peaks example");
    println!("------------------------");
    let x_multi = Array1::linspace(0.0, 10.0, 200);
    let mut y_multi = Array1::zeros(200);

    // Three Gaussians + baseline
    // G1: amplitude=3.0, center=2.0, sigma=0.5
    // G2: amplitude=4.0, center=5.0, sigma=0.8
    // G3: amplitude=2.5, center=8.0, sigma=0.6
    // baseline=0.5
    for i in 0..200 {
        let x = x_multi[i];
        let noise = rng.gen_range(-0.2..0.2);

        // First Gaussian
        let arg1 = (x - 2.0) / 0.5;
        let g1 = 3.0 * (-0.5_f64 * arg1 * arg1).exp();

        // Second Gaussian
        let arg2 = (x - 5.0) / 0.8;
        let g2 = 4.0 * (-0.5_f64 * arg2 * arg2).exp();

        // Third Gaussian
        let arg3 = (x - 8.0) / 0.6;
        let g3 = 2.5 * (-0.5_f64 * arg3 * arg3).exp();

        y_multi[i] = g1 + g2 + g3 + 0.5 + noise;
    }

    // Create parameters for a multi-peak model
    let mut parameters = Parameters::new();

    // Add parameters for three Gaussians
    parameters.add_param_with_bounds("g1_amplitude", 1.0, 0.0, 10.0)?;
    parameters.add_param_with_bounds("g1_center", 1.0, 0.0, 5.0)?;
    parameters.add_param_with_bounds("g1_sigma", 1.0, 0.1, 2.0)?;

    parameters.add_param_with_bounds("g2_amplitude", 1.0, 0.0, 10.0)?;
    parameters.add_param_with_bounds("g2_center", 5.0, 3.0, 7.0)?;
    parameters.add_param_with_bounds("g2_sigma", 1.0, 0.1, 2.0)?;

    parameters.add_param_with_bounds("g3_amplitude", 1.0, 0.0, 10.0)?;
    parameters.add_param_with_bounds("g3_center", 9.0, 7.0, 10.0)?;
    parameters.add_param_with_bounds("g3_sigma", 1.0, 0.1, 2.0)?;

    parameters.add_param("baseline", 0.0)?;

    // Create a custom model using BaseModel
    let model = BaseModel::new(parameters, |params, x| {
        // Get parameters for each Gaussian
        let g1_amp = params.get("g1_amplitude").unwrap().value();
        let g1_center = params.get("g1_center").unwrap().value();
        let g1_sigma = params.get("g1_sigma").unwrap().value();

        let g2_amp = params.get("g2_amplitude").unwrap().value();
        let g2_center = params.get("g2_center").unwrap().value();
        let g2_sigma = params.get("g2_sigma").unwrap().value();

        let g3_amp = params.get("g3_amplitude").unwrap().value();
        let g3_center = params.get("g3_center").unwrap().value();
        let g3_sigma = params.get("g3_sigma").unwrap().value();

        let baseline = params.get("baseline").unwrap().value();

        // Calculate model
        let result = x
            .iter()
            .map(|&x_val| {
                // First Gaussian
                let arg1 = (x_val - g1_center) / g1_sigma;
                let g1 = g1_amp * (-0.5 * arg1 * arg1).exp();

                // Second Gaussian
                let arg2 = (x_val - g2_center) / g2_sigma;
                let g2 = g2_amp * (-0.5 * arg2 * arg2).exp();

                // Third Gaussian
                let arg3 = (x_val - g3_center) / g3_sigma;
                let g3 = g3_amp * (-0.5 * arg3 * arg3).exp();

                // Sum of all peaks plus baseline
                g1 + g2 + g3 + baseline
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(result))
    });

    let mut custom_model = model;
    let multi_result = fit(&mut custom_model, x_multi.clone(), y_multi.clone())?;

    // Output results
    println!("Fit success: {}", multi_result.success);
    println!("Parameters:");
    println!(
        "  G1 amplitude = {:.3}",
        custom_model
            .parameters()
            .get("g1_amplitude")
            .unwrap()
            .value()
    );
    println!(
        "  G1 center = {:.3}",
        custom_model.parameters().get("g1_center").unwrap().value()
    );
    println!(
        "  G1 sigma = {:.3}",
        custom_model.parameters().get("g1_sigma").unwrap().value()
    );
    println!();
    println!(
        "  G2 amplitude = {:.3}",
        custom_model
            .parameters()
            .get("g2_amplitude")
            .unwrap()
            .value()
    );
    println!(
        "  G2 center = {:.3}",
        custom_model.parameters().get("g2_center").unwrap().value()
    );
    println!(
        "  G2 sigma = {:.3}",
        custom_model.parameters().get("g2_sigma").unwrap().value()
    );
    println!();
    println!(
        "  G3 amplitude = {:.3}",
        custom_model
            .parameters()
            .get("g3_amplitude")
            .unwrap()
            .value()
    );
    println!(
        "  G3 center = {:.3}",
        custom_model.parameters().get("g3_center").unwrap().value()
    );
    println!(
        "  G3 sigma = {:.3}",
        custom_model.parameters().get("g3_sigma").unwrap().value()
    );
    println!();
    println!(
        "  baseline = {:.3}",
        custom_model.parameters().get("baseline").unwrap().value()
    );
    println!("Cost (residual sum): {:.6}", multi_result.cost);

    // Standard errors for all parameters
    println!("Standard errors:");
    let param_names = [
        "g1_amplitude",
        "g1_center",
        "g1_sigma",
        "g2_amplitude",
        "g2_center",
        "g2_sigma",
        "g3_amplitude",
        "g3_center",
        "g3_sigma",
        "baseline",
    ];
    for name in param_names.iter() {
        println!(
            "  {} = {:.3}",
            name,
            multi_result
                .standard_errors
                .get(&name.to_string())
                .unwrap_or(&0.0)
        );
    }

    println!("\nExample completed successfully!");
    Ok(())
}
