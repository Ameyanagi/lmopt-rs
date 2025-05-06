use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::GlobalOptimizer;
use lmopt_rs::parameters::Parameters;
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use lmopt_rs::uncertainty::{confidence_intervals, ConfidenceInterval};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create a simple quadratic model: f(x) = a*x^2 + b*x + c
    // with some synthetic data
    println!("Confidence Interval Example");
    println!("==========================\n");

    // Generate synthetic data
    let true_a = 2.0;
    let true_b = 5.0;
    let true_c = 1.0;

    let x_values: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
    let mut y_values = Vec::new();

    // Add some noise to the data
    let mut rng = rand::thread_rng();
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0, 2.0).unwrap(); // Mean 0, standard deviation 2

    for &x in &x_values {
        let y_true = true_a * x * x + true_b * x + true_c;
        let noise = normal.sample(&mut rng);
        y_values.push(y_true + noise);
    }

    // Create parameter set
    let mut params = Parameters::new();
    params.add_param("a", 1.0)?; // Initial guess
    params.add_param("b", 1.0)?;
    params.add_param("c", 0.0)?;

    // Define the problem
    let mut quadratic_problem = QuadraticProblem::new(x_values.clone(), y_values.clone(), params);

    // Create the optimization adapter
    let adapter = problem_from_parameter_problem(&quadratic_problem);

    // Create the optimizer with default configuration
    let optimizer = lmopt_rs::lm::LevenbergMarquardt::with_default_config();

    // Get initial parameters
    let initial_params = quadratic_problem.parameters_to_array()?;

    // Run the optimization
    println!("\nRunning optimization...");
    let result = optimizer.minimize(&adapter, initial_params)?;

    // Update the model's parameters with the optimized values
    quadratic_problem.update_parameters_from_array(&result.params)?;

    // Print optimized parameters
    println!(
        "\nOptimization completed in {} iterations",
        result.iterations
    );
    println!("Final cost: {:.6}", result.cost);

    // Print the fitted parameters
    println!("\nOptimized parameters:");
    println!(
        "a = {:.4}",
        quadratic_problem.parameters().get("a").unwrap().value()
    );
    println!(
        "b = {:.4}",
        quadratic_problem.parameters().get("b").unwrap().value()
    );
    println!(
        "c = {:.4}",
        quadratic_problem.parameters().get("c").unwrap().value()
    );

    // Calculate confidence intervals using standard covariance method
    println!("\nEstimating confidence intervals from covariance matrix...");

    // Calculate the Jacobian at the optimum
    let jacobian = quadratic_problem.jacobian_with_parameters()?;

    // Get covariance matrix with the right function signature
    // Using covariance_matrix(jacobian, chisqr, ndata, nvarys)
    let chisqr = result.cost;
    let ndata = quadratic_problem.residual_count();
    let nvarys = quadratic_problem.parameters().varying().len();

    let covar = lmopt_rs::uncertainty::covariance_matrix(&jacobian, chisqr, ndata, nvarys)?;

    // Calculate standard confidence intervals for 1, 2, and 3 sigma levels
    let sigmas = &[1.0, 2.0, 3.0];
    let covar_intervals =
        confidence_intervals(quadratic_problem.parameters(), &covar, sigmas, None)?;

    // Print confidence intervals
    println!("\nConfidence intervals (covariance method):");
    print_confidence_intervals(&covar_intervals);

    // For profile likelihood intervals, we would need to implement using a custom approach
    // but we'll comment it out for now since it's not directly exposed in the API
    /*
    println!("\nEstimating confidence intervals using profile likelihood method...");
    println!("(This may take some time as it requires multiple optimizations)\n");

    // Use 68%, 95%, and 99% confidence levels
    let confidence_levels = &[0.6827, 0.9545, 0.9973];

    // This will require using our own implementation since profile_likelihood_intervals
    // isn't exposed in the public API
    */

    // Compare the two methods
    println!("\nNote on confidence interval methods:");
    println!("For linear problems with normally distributed errors,");
    println!("covariance-based confidence intervals are usually accurate.");
    println!("For nonlinear problems or non-Gaussian errors,");
    println!("profile likelihood methods are generally more accurate but");
    println!("require multiple optimizations.");

    Ok(())
}

// Helper function to print confidence intervals
fn print_confidence_intervals(intervals: &HashMap<String, Vec<ConfidenceInterval>>) {
    for (param_name, param_intervals) in intervals {
        println!("\nParameter: {}", param_name);

        for interval in param_intervals {
            let confidence = interval.probability * 100.0;
            println!(
                "{:.1}% confidence interval: [{:.4}, {:.4}]",
                confidence, interval.lower, interval.upper
            );
        }
    }
}

// Define a simple quadratic model problem
#[derive(Clone)]
struct QuadraticProblem {
    x: Vec<f64>,
    y: Vec<f64>,
    parameters: Parameters,
}

impl QuadraticProblem {
    fn new(x: Vec<f64>, y: Vec<f64>, parameters: Parameters) -> Self {
        Self { x, y, parameters }
    }
}

impl ParameterProblem for QuadraticProblem {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        let a = self
            .parameters
            .get("a")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'a' not found".to_string()))?
            .value();
        let b = self
            .parameters
            .get("b")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'b' not found".to_string()))?
            .value();
        let c = self
            .parameters
            .get("c")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'c' not found".to_string()))?
            .value();

        // Calculate residuals: y_observed - y_model
        let mut residuals = Array1::zeros(self.x.len());

        for (i, &x) in self.x.iter().enumerate() {
            let y_model = a * x * x + b * x + c;
            residuals[i] = self.y[i] - y_model;
        }

        Ok(residuals)
    }

    fn jacobian_with_parameters(&self) -> Result<Array2<f64>> {
        let _a = self
            .parameters
            .get("a")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'a' not found".to_string()))?
            .value();
        let _b = self
            .parameters
            .get("b")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'b' not found".to_string()))?
            .value();
        let _c = self
            .parameters
            .get("c")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'c' not found".to_string()))?
            .value();

        // Calculate Jacobian matrix
        // Each row corresponds to a data point
        // Each column corresponds to a parameter: [∂r/∂a, ∂r/∂b, ∂r/∂c]
        let mut jacobian = Array2::zeros((self.x.len(), 3));

        for (i, &x) in self.x.iter().enumerate() {
            // ∂r/∂a = -x²
            jacobian[[i, 0]] = -x * x;

            // ∂r/∂b = -x
            jacobian[[i, 1]] = -x;

            // ∂r/∂c = -1
            jacobian[[i, 2]] = -1.0;
        }

        Ok(jacobian)
    }

    fn residual_count(&self) -> usize {
        self.x.len()
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

// Implement GlobalOptimizer for QuadraticProblem to enable profile likelihood methods
impl GlobalOptimizer for QuadraticProblem {
    fn optimize<P: lmopt_rs::problem::Problem>(
        &self,
        _problem: &P,
        _bounds: &[(f64, f64)],
        _max_iterations: usize,
        _max_no_improvement: usize,
        _tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        // This is just a stub implementation for the example
        // In a real implementation, we would use the HybridGlobal or another optimizer
        Err(LmOptError::NotImplemented(
            "Custom global optimization not implemented for this example".to_string(),
        ))
    }

    fn optimize_param_problem(
        &mut self,
        _max_iterations: usize,
        _max_no_improvement: usize,
        _tol: f64,
    ) -> Result<lmopt_rs::global_opt::GlobalOptResult> {
        // In a real implementation, we would implement this properly
        // For now, just use LevenbergMarquardt from lmopt_rs crate
        let adapter = problem_from_parameter_problem(self);
        let optimizer = lmopt_rs::lm::LevenbergMarquardt::with_default_config();
        let initial_params = self.parameters_to_array()?;

        let result = optimizer.minimize(&adapter, initial_params)?;

        // Convert to a GlobalOptResult
        Ok(lmopt_rs::global_opt::GlobalOptResult {
            params: result.params.clone(),
            cost: result.cost,
            iterations: result.iterations,
            func_evals: result.iterations, // Using iterations as a proxy since LmResult doesn't have func_evals
            success: result.success,
            message: result.message.clone(),
            local_result: Some(result),
        })
    }
}
