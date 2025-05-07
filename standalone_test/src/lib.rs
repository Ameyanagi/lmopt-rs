//! This is a simplified version of the uncertainty propagation module
//! for testing.

use approx::assert_relative_eq;
use ndarray::Array1;
use std::collections::HashMap;

/// Result of a Monte Carlo uncertainty analysis.
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
pub fn propagate_uncertainty<F>(mc_result: &MonteCarloResult, func: F) -> (Array1<f64>, f64, f64)
where
    F: Fn(&HashMap<String, f64>) -> f64,
{
    // Calculate the derived quantity for each parameter set
    let derived_values: Vec<f64> = mc_result
        .parameter_sets
        .iter()
        .map(|params| func(params))
        .collect();

    // Convert to ndarray for easier operations
    let derived_array = Array1::from_vec(derived_values.clone());

    // Calculate mean
    let mean = derived_values.iter().sum::<f64>() / derived_values.len() as f64;

    // Calculate standard deviation
    let var = derived_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / derived_values.len() as f64;
    let std_dev = var.sqrt();

    (derived_array, mean, std_dev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    fn create_test_mc_result() -> MonteCarloResult {
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

        // Calculate means and standard deviations
        let mut means = HashMap::new();
        let mut stds = HashMap::new();

        // Calculate sums
        let mut a_sum = 0.0;
        let mut b_sum = 0.0;

        for params in &parameter_sets {
            a_sum += params["a"];
            b_sum += params["b"];
        }

        let a_mean = a_sum / parameter_sets.len() as f64;
        let b_mean = b_sum / parameter_sets.len() as f64;

        means.insert("a".to_string(), a_mean);
        means.insert("b".to_string(), b_mean);

        // Calculate variances
        let mut a_var_sum = 0.0;
        let mut b_var_sum = 0.0;

        for params in &parameter_sets {
            a_var_sum += (params["a"] - a_mean).powi(2);
            b_var_sum += (params["b"] - b_mean).powi(2);
        }

        let a_std = (a_var_sum / parameter_sets.len() as f64).sqrt();
        let b_std = (b_var_sum / parameter_sets.len() as f64).sqrt();

        stds.insert("a".to_string(), a_std);
        stds.insert("b".to_string(), b_std);

        MonteCarloResult {
            parameter_sets,
            means,
            stds,
        }
    }

    #[test]
    fn test_propagate_uncertainty_linear() {
        let mc_result = create_test_mc_result();

        // Define a linear function: 2*a + 3*b
        let linear_func = |params: &HashMap<String, f64>| 2.0 * params["a"] + 3.0 * params["b"];

        // Propagate uncertainty
        let (_, linear_mean, linear_std) = propagate_uncertainty(&mc_result, linear_func);

        // Expected mean: 2*E[a] + 3*E[b] = 2*2 + 3*5 = 4 + 15 = 19
        assert_relative_eq!(linear_mean, 19.0, epsilon = 0.5);

        // Expected std: sqrt((2*std_a)^2 + (3*std_b)^2)
        // = sqrt(4*0.2^2 + 9*0.5^2) = sqrt(0.16 + 2.25) = sqrt(2.41) ≈ 1.55
        assert_relative_eq!(linear_std, 1.55, epsilon = 0.2);
    }

    #[test]
    fn test_propagate_uncertainty_nonlinear() {
        let mc_result = create_test_mc_result();

        // Define a nonlinear function: a*b^2
        let nonlinear_func = |params: &HashMap<String, f64>| params["a"] * params["b"].powi(2);

        // Propagate uncertainty
        let (_, nonlinear_mean, nonlinear_std) = propagate_uncertainty(&mc_result, nonlinear_func);

        // For nonlinear function, we expect:
        // E[a*b^2] ≈ E[a]*E[b^2] = 2*(5^2 + 0.5^2) = 2*(25 + 0.25) = 2*25.25 = 50.5
        // But there will be some difference due to correlation and non-linearity
        assert_relative_eq!(nonlinear_mean, 50.5, epsilon = 5.0);
    }

    #[test]
    fn test_model_prediction_uncertainty() {
        // Create a more complex parameter set with quadratic model parameters (a, b, c)
        let mut parameter_sets = Vec::new();

        // Use a fixed seed for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let normal_a = Normal::new(2.0, 0.2).unwrap(); // Mean 2.0, std 0.2
        let normal_b = Normal::new(5.0, 0.5).unwrap(); // Mean 5.0, std 0.5
        let normal_c = Normal::new(1.0, 0.1).unwrap(); // Mean 1.0, std 0.1

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

        // Calculate means
        let mut means = HashMap::new();
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
        let mut stds = HashMap::new();
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

        // Create Monte Carlo result
        let mc_result = MonteCarloResult {
            parameter_sets,
            means,
            stds,
        };

        // Define x points where we want to evaluate the model
        let x_values = vec![0.0, 1.0, 3.0];

        // Store results for each x value
        let mut results = Vec::new();

        // For each x point, evaluate model function
        for &x in &x_values {
            // Define quadratic model function f(x) = a*x^2 + b*x + c
            let model_func = move |params: &HashMap<String, f64>| {
                let a = params["a"];
                let b = params["b"];
                let c = params["c"];
                a * x * x + b * x + c
            };

            // Propagate uncertainty
            let (_, mean, std_dev) = propagate_uncertainty(&mc_result, model_func);
            results.push((x, mean, std_dev));
        }

        // For x = 0, expect just c
        let (x0, mean0, std0) = results[0];
        assert_eq!(x0, 0.0);
        assert_relative_eq!(mean0, c_mean, epsilon = 0.1);
        assert_relative_eq!(std0, c_std, epsilon = 0.1);

        // For x = 1, expect a + b + c
        let (x1, mean1, std1) = results[1];
        assert_eq!(x1, 1.0);
        assert_relative_eq!(mean1, a_mean + b_mean + c_mean, epsilon = 0.3);

        // For uncertainty at x=3, should be greater than at x=0 and x=1
        let (x3, mean3, std3) = results[2];
        assert_eq!(x3, 3.0);
        assert!(
            std3 > std0,
            "Uncertainty at x=3 should be greater than at x=0"
        );
        assert!(
            std3 > std1,
            "Uncertainty at x=3 should be greater than at x=1"
        );
    }
}
