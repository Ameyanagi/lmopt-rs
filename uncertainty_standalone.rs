//! Completely standalone test for uncertainty propagation.
//!
//! This is a simplified implementation of uncertainty propagation
//! extracted from the main codebase for testing.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Test result of a Monte Carlo uncertainty analysis.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// The parameter sets from all Monte Carlo iterations
    pub parameter_sets: Vec<HashMap<String, f64>>,

    /// Mean parameter values across all simulations
    pub means: HashMap<String, f64>,

    /// Standard deviations of parameters across all simulations
    pub stds: HashMap<String, f64>,
}

/// Propagate parameter uncertainties to a derived quantity.
///
/// This function uses Monte Carlo parameter sets to calculate the distribution
/// of a derived quantity, which is a function of the parameters.
pub fn propagate_uncertainty<F>(mc_result: &MonteCarloResult, func: F) -> (Vec<f64>, f64, f64)
where
    F: Fn(&HashMap<String, f64>) -> f64,
{
    // Calculate the derived quantity for each parameter set
    let derived_values: Vec<f64> = mc_result
        .parameter_sets
        .iter()
        .map(|params| func(params))
        .collect();

    // Calculate mean
    let mean = derived_values.iter().sum::<f64>() / derived_values.len() as f64;

    // Calculate standard deviation
    let var = derived_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / derived_values.len() as f64;
    let std_dev = var.sqrt();

    (derived_values, mean, std_dev)
}

fn test_propagate_uncertainty() {
    // Create a sample MonteCarloResult
    let mut parameter_sets = Vec::new();

    // Use a fixed seed for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let normal_a = Normal::new(2.0, 0.2).unwrap(); // Mean 2.0, std 0.2
    let normal_b = Normal::new(5.0, 0.5).unwrap(); // Mean 5.0, std 0.5

    // Generate 1000 parameter sets
    for _ in 0..1000 {
        let a = normal_a.sample(&mut rng);
        let b = normal_b.sample(&mut rng);

        let mut params = HashMap::new();
        params.insert("a".to_string(), a);
        params.insert("b".to_string(), b);

        parameter_sets.push(params);
    }

    // Calculate means and standard deviations for the parameters
    let mut a_sum = 0.0;
    let mut b_sum = 0.0;

    for params in &parameter_sets {
        a_sum += params["a"];
        b_sum += params["b"];
    }

    let a_mean = a_sum / parameter_sets.len() as f64;
    let b_mean = b_sum / parameter_sets.len() as f64;

    let mut a_var_sum = 0.0;
    let mut b_var_sum = 0.0;

    for params in &parameter_sets {
        a_var_sum += (params["a"] - a_mean).powi(2);
        b_var_sum += (params["b"] - b_mean).powi(2);
    }

    let a_std = (a_var_sum / parameter_sets.len() as f64).sqrt();
    let b_std = (b_var_sum / parameter_sets.len() as f64).sqrt();

    let mut means = HashMap::new();
    means.insert("a".to_string(), a_mean);
    means.insert("b".to_string(), b_mean);

    let mut stds = HashMap::new();
    stds.insert("a".to_string(), a_std);
    stds.insert("b".to_string(), b_std);

    // Create the Monte Carlo result
    let mc_result = MonteCarloResult {
        parameter_sets,
        means,
        stds,
    };

    // Define functions for derived quantities
    // 1. Linear combination: 2*a + 3*b
    let linear_func = |params: &HashMap<String, f64>| 2.0 * params["a"] + 3.0 * params["b"];

    // 2. Nonlinear function: a*b^2
    let nonlinear_func = |params: &HashMap<String, f64>| params["a"] * params["b"].powi(2);

    // Test linear propagation
    let (linear_values, linear_mean, linear_std) = propagate_uncertainty(&mc_result, linear_func);

    // Expected mean for linear function: 2*E[a] + 3*E[b] = 2*2 + 3*5 = 4 + 15 = 19
    assert!(
        (linear_mean - 19.0).abs() < 0.5,
        "Linear mean should be close to 19"
    );

    // Expected std for linear function: sqrt((2*std_a)^2 + (3*std_b)^2)
    // = sqrt(4*0.2^2 + 9*0.5^2) = sqrt(0.16 + 2.25) = sqrt(2.41) ≈ 1.55
    assert!(
        (linear_std - 1.55).abs() < 0.2,
        "Linear std should be close to 1.55"
    );

    // Test nonlinear propagation
    let (nonlinear_values, nonlinear_mean, nonlinear_std) =
        propagate_uncertainty(&mc_result, nonlinear_func);

    // For nonlinear function, we expect:
    // E[a*b^2] ≈ E[a]*E[b^2] = 2*(5^2 + 0.5^2) = 2*(25 + 0.25) = 2*25.25 = 50.5
    // But there will be some difference due to correlation and non-linearity
    println!("Nonlinear mean: {}", nonlinear_mean);
    assert!(
        (nonlinear_mean - 50.5).abs() < 5.0,
        "Nonlinear mean should be approximately 50.5"
    );
}

fn test_uncertainty_propagation_to_model_prediction() {
    // Create parameter distribution
    let mut parameter_sets = Vec::new();

    // Use a fixed seed
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Normal distributions for parameters
    let normal_a = Normal::new(2.0, 0.2).unwrap();
    let normal_b = Normal::new(5.0, 0.5).unwrap();
    let normal_c = Normal::new(1.0, 0.1).unwrap();

    // Generate parameters
    for _ in 0..1000 {
        let a = normal_a.sample(&mut rng);
        let b = normal_b.sample(&mut rng);
        let c = normal_c.sample(&mut rng);

        let mut params = HashMap::new();
        params.insert("a".to_string(), a);
        params.insert("b".to_string(), b);
        params.insert("c".to_string(), c);

        parameter_sets.push(params);
    }

    // Calculate means and standard deviations
    let mut means = HashMap::new();
    let mut stds = HashMap::new();

    // Calculate means
    let mut a_sum = 0.0;
    let mut b_sum = 0.0;
    let mut c_sum = 0.0;

    for params in &parameter_sets {
        a_sum += params["a"];
        b_sum += params["b"];
        c_sum += params["c"];
    }

    let a_mean = a_sum / parameter_sets.len() as f64;
    let b_mean = b_sum / parameter_sets.len() as f64;
    let c_mean = c_sum / parameter_sets.len() as f64;

    means.insert("a".to_string(), a_mean);
    means.insert("b".to_string(), b_mean);
    means.insert("c".to_string(), c_mean);

    // Calculate standard deviations
    let mut a_var_sum = 0.0;
    let mut b_var_sum = 0.0;
    let mut c_var_sum = 0.0;

    for params in &parameter_sets {
        a_var_sum += (params["a"] - a_mean).powi(2);
        b_var_sum += (params["b"] - b_mean).powi(2);
        c_var_sum += (params["c"] - c_mean).powi(2);
    }

    let a_std = (a_var_sum / parameter_sets.len() as f64).sqrt();
    let b_std = (b_var_sum / parameter_sets.len() as f64).sqrt();
    let c_std = (c_var_sum / parameter_sets.len() as f64).sqrt();

    stds.insert("a".to_string(), a_std);
    stds.insert("b".to_string(), b_std);
    stds.insert("c".to_string(), c_std);

    // Create a dummy MonteCarloResult
    let mc_result = MonteCarloResult {
        parameter_sets,
        means,
        stds,
    };

    // Define x points where we want to evaluate the model
    let x_values = vec![-3.0, -1.0, 0.0, 1.0, 3.0];

    // For each x value, create a function to evaluate the model and propagate uncertainty
    let mut predictions = HashMap::new();

    for &x in &x_values {
        // Define a function to evaluate the quadratic at this x
        let model_at_x = move |params: &HashMap<String, f64>| {
            let a = params["a"];
            let b = params["b"];
            let c = params["c"];
            a * x * x + b * x + c
        };

        // Propagate uncertainty
        let (values, mean, std_dev) = propagate_uncertainty(&mc_result, model_at_x);
        predictions.insert(x, (mean, std_dev));
    }

    // Check predictions at x = 0
    // Should be just the value of parameter c
    let pred_at_0 = &predictions[&0.0];
    assert!((pred_at_0.0 - 1.0).abs() < 0.2); // Mean should be close to c=1.0
    assert!((pred_at_0.1 - 0.1).abs() < 0.05); // Std dev should be close to std_c=0.1

    // Check predictions at x = 1
    // Should be a + b + c = 2 + 5 + 1 = 8
    let pred_at_1 = &predictions[&1.0];
    assert!((pred_at_1.0 - 8.0).abs() < 0.5);

    // Check predictions at x = 3
    // Should be 9*a + 3*b + c = 9*2 + 3*5 + 1 = 18 + 15 + 1 = 34
    let pred_at_3 = &predictions[&3.0];
    assert!((pred_at_3.0 - 34.0).abs() < 1.0);

    // Check that uncertainty increases with the magnitude of x
    // This is because the coefficient of x^2 (a) has uncertainty
    assert!(
        predictions[&0.0].1 < predictions[&1.0].1 && predictions[&1.0].1 < predictions[&3.0].1,
        "Uncertainty should increase with |x|"
    );
}

fn main() {
    println!("Running uncertainty propagation tests...");
    test_propagate_uncertainty();
    test_uncertainty_propagation_to_model_prediction();
    println!("All tests passed!");
}
