use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::GlobalOptimizer;
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use lmopt_rs::uncertainty::{
    monte_carlo_covariance, monte_carlo_refit, MonteCarloResult, UncertaintyCalculator,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create a simple quadratic model: f(x) = a*x^2 + b*x + c
    // with some synthetic data
    println!("Monte Carlo Uncertainty Analysis Example");
    println!("======================================\n");

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

    // Create the optimizer
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

    // Calculate uncertainty using standard methods
    println!("\nCalculating uncertainty using covariance matrix...");
    let ndata = quadratic_problem.residual_count();
    let nvarys = quadratic_problem.parameters().varying().len();

    let uncertainty = UncertaintyCalculator::new(ndata, nvarys, result.cost);

    // Calculate the Jacobian at the optimum
    let jacobian = quadratic_problem.jacobian_with_parameters()?;

    // Compute covariance matrix
    let covar = lmopt_rs::uncertainty::covariance_matrix(&jacobian, result.cost, ndata, nvarys)?;

    // Calculate standard errors
    let std_errors = lmopt_rs::uncertainty::standard_errors(&covar, quadratic_problem.parameters());

    // Print standard errors
    println!("\nStandard errors from covariance matrix:");
    for (param_name, std_error) in std_errors {
        println!("{}: ±{:.4}", param_name, std_error);
    }

    // Run Monte Carlo analysis with covariance matrix
    println!("\nRunning Monte Carlo uncertainty analysis (covariance method)...");

    // Use a seeded RNG for reproducibility
    let mut mc_rng = ChaCha8Rng::seed_from_u64(42);
    let mc_results = monte_carlo_covariance(
        quadratic_problem.parameters(),
        &covar,
        1000,                // Number of samples
        &[0.68, 0.95, 0.99], // Confidence levels
        &mut mc_rng,
    )?;

    // Print Monte Carlo results
    println!("\nMonte Carlo results (parameter distributions):");
    print_monte_carlo_results(&mc_results);

    // Calculate derived quantity from Monte Carlo samples
    println!("\nPropagating uncertainty to derived quantity...");
    println!("Example: Value at x=2.0, i.e., a*4 + b*2 + c");

    // Define function to calculate derived quantity
    let derived_func = |params: &HashMap<String, f64>| {
        let a = params["a"];
        let b = params["b"];
        let c = params["c"];
        a * 4.0 + b * 2.0 + c // Value of the quadratic at x=2
    };

    // Propagate uncertainty to derived quantity
    let (values, mean, std_dev, median, percentiles) = lmopt_rs::uncertainty::propagate_uncertainty(
        &mc_results,
        derived_func,
        &[0.68, 0.95, 0.99],
    );

    println!("\nPropagated uncertainty for f(2.0):");
    println!("Mean value: {:.4}", mean);
    println!("Standard deviation: {:.4}", std_dev);
    println!("Median value: {:.4}", median);

    println!("\nConfidence intervals:");
    for (p, (lower, upper)) in percentiles {
        println!("{:.1}% CI: [{:.4}, {:.4}]", p * 100.0, lower, upper);
    }

    // For monte_carlo_refit we need to implement GlobalOptimizer trait
    // Note that this is commented out for now as the implementation is a stub
    /*
    // Run Monte Carlo with refitting (this would be more accurate but slower)
    println!("\nRunning Monte Carlo with refitting...");
    // Get residuals for the best fit
    let residuals = quadratic_problem.eval_with_parameters()?;

    // Run the Monte Carlo refitting analysis
    let mc_refit_results = monte_carlo_refit(
        &mut quadratic_problem,
        quadratic_problem.parameters(),
        &residuals,
        100,  // Fewer samples since this is more computationally intensive
        &[0.68, 0.95],
        &mut mc_rng,
    )?;

    // Print results
    println!("\nMonte Carlo refit results:");
    print_monte_carlo_results(&mc_refit_results);
    */

    // Compare the two methods
    println!("\nComparison of Monte Carlo methods:");
    println!("The covariance-based Monte Carlo is faster but assumes a");
    println!("multivariate normal parameter distribution.\n");
    println!("The refitting-based Monte Carlo is more computationally intensive");
    println!("but can capture non-linear parameter dependencies and non-Gaussian distributions.");

    Ok(())
}

// Helper function to print Monte Carlo results
fn print_monte_carlo_results(results: &MonteCarloResult) {
    for (param_name, mean) in &results.means {
        println!("\nParameter: {}", param_name);
        println!("Mean: {:.4}", mean);
        println!("Std Dev: {:.4}", results.stds[param_name]);
        println!("Median: {:.4}", results.medians[param_name]);

        println!("Confidence intervals:");
        for (p, (lower, upper)) in &results.percentiles[param_name] {
            println!("{:.1}% CI: [{:.4}, {:.4}]", p * 100.0, lower, upper);
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

    // Method to update the data for Monte Carlo refitting
    fn set_data(&mut self, new_y: &Array1<f64>) -> Result<()> {
        if new_y.len() != self.y.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} data points, got {}",
                self.y.len(),
                new_y.len()
            )));
        }

        // Update the y values with the new synthetic data
        for i in 0..self.y.len() {
            self.y[i] = new_y[i];
        }

        Ok(())
    }

    // Additional method for evaluating model predictions (not part of the trait)
    fn evaluate_model(&self, params: &Parameters) -> Result<Array1<f64>> {
        let a = params
            .get("a")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'a' not found".to_string()))?
            .value();
        let b = params
            .get("b")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'b' not found".to_string()))?
            .value();
        let c = params
            .get("c")
            .ok_or_else(|| LmOptError::ParameterError("Parameter 'c' not found".to_string()))?
            .value();

        // Calculate model predictions
        let mut predictions = Array1::zeros(self.x.len());

        for (i, &x) in self.x.iter().enumerate() {
            predictions[i] = a * x * x + b * x + c;
        }

        Ok(predictions)
    }
}

// Implement GlobalOptimizer for QuadraticProblem
// This is necessary for monte_carlo_refit but commented out since we're not using it
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
