//! Example demonstrating the use of Parameter system with Levenberg-Marquardt optimization
//!
//! This example shows how to use the Parameter system to define and solve a nonlinear
//! least-squares problem with parameter bounds, constraints, and expressions.

use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::lm::LevenbergMarquardt;
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use ndarray::{array, Array1, Array2};

/// A Gaussian peak model: f(x) = a * exp(-(x - b)^2 / (2 * c^2))
struct GaussianModel {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    parameters: Parameters,
}

impl GaussianModel {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        assert_eq!(
            x_data.len(),
            y_data.len(),
            "x and y data must have the same length"
        );

        // Create the parameters:
        // - a: amplitude (positive)
        // - b: center
        // - c: width (positive)
        let mut parameters = Parameters::new();
        parameters
            .add_param_with_bounds("a", 1.0, 0.0, f64::INFINITY)
            .unwrap();
        parameters.add_param("b", 0.0).unwrap();
        parameters
            .add_param_with_bounds("c", 1.0, 0.01, f64::INFINITY)
            .unwrap();

        // Add a derived parameter for FWHM (Full Width at Half Maximum)
        // FWHM = 2.355 * c for Gaussian
        parameters
            .add_param_with_expr("fwhm", 2.355, "2.355 * c")
            .unwrap();

        Self {
            x_data,
            y_data,
            parameters,
        }
    }

    /// Evaluate the Gaussian function at a point
    fn gaussian(x: f64, a: f64, b: f64, c: f64) -> f64 {
        a * (-((x - b).powi(2)) / (2.0 * c.powi(2))).exp()
    }

    /// Print the fit results
    fn print_results(&self) {
        println!("Fit Results:");
        println!(
            "  Amplitude (a): {:.6}",
            self.parameters.get("a").unwrap().value()
        );
        println!(
            "  Center (b): {:.6}",
            self.parameters.get("b").unwrap().value()
        );
        println!(
            "  Width (c): {:.6}",
            self.parameters.get("c").unwrap().value()
        );
        println!(
            "  FWHM: {:.6}",
            self.parameters.get("fwhm").unwrap().value()
        );

        // Calculate goodness of fit
        let residuals = self.eval_with_parameters().unwrap();
        let ssr = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
        let mean_y = self.y_data.iter().sum::<f64>() / self.y_data.len() as f64;
        let sst = self
            .y_data
            .iter()
            .map(|y| (y - mean_y).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - ssr / sst;

        println!("Goodness of fit:");
        println!("  Sum of squared residuals: {:.6}", ssr);
        println!("  R-squared: {:.6}", r_squared);
    }
}

impl ParameterProblem for GaussianModel {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        // Get parameter values
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

        // Calculate residuals
        let residuals = self
            .x_data
            .iter()
            .zip(self.y_data.iter())
            .map(|(x, y)| Self::gaussian(*x, a, b, c) - y)
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(residuals))
    }

    fn residual_count(&self) -> usize {
        self.x_data.len()
    }

    // We'll use auto-differentiation here (via finite differences)
    fn has_custom_jacobian(&self) -> bool {
        false
    }
}

fn main() -> Result<()> {
    // Generate some synthetic data with noise
    let x = Array1::linspace(-5.0, 5.0, 100);

    // True parameters: a = 3.0, b = 1.0, c = 0.5
    let true_a = 3.0;
    let true_b = 1.0;
    let true_c = 0.5;

    // Generate data with some noise
    let mut rng = rand::thread_rng();
    let noise_level = 0.1;
    let y = x.mapv(|x_val| {
        GaussianModel::gaussian(x_val, true_a, true_b, true_c)
            + noise_level * (rand::random::<f64>() - 0.5)
    });

    // Create the model
    let mut model = GaussianModel::new(x, y);

    // Set initial guesses away from the true values
    model
        .parameters_mut()
        .get_mut("a")
        .unwrap()
        .set_value(1.0)
        .unwrap();
    model
        .parameters_mut()
        .get_mut("b")
        .unwrap()
        .set_value(0.0)
        .unwrap();
    model
        .parameters_mut()
        .get_mut("c")
        .unwrap()
        .set_value(1.0)
        .unwrap();

    // Print initial parameters
    println!("Initial parameters:");
    println!(
        "  Amplitude (a): {:.6}",
        model.parameters().get("a").unwrap().value()
    );
    println!(
        "  Center (b): {:.6}",
        model.parameters().get("b").unwrap().value()
    );
    println!(
        "  Width (c): {:.6}",
        model.parameters().get("c").unwrap().value()
    );
    println!(
        "  FWHM: {:.6}",
        model.parameters().get("fwhm").unwrap().value()
    );

    // Create the optimization adapter
    let adapter = problem_from_parameter_problem(&model);

    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();

    // Get initial parameters
    let initial_params = model.parameters_to_array()?;

    // Run the optimization
    println!("\nRunning optimization...");
    let result = optimizer.minimize(&adapter, initial_params)?;

    // Update the model's parameters with the optimized values
    model.update_parameters_from_array(&result.params)?;

    // Print optimization results
    println!(
        "\nOptimization completed in {} iterations",
        result.iterations
    );
    println!("Final cost: {:.6}", result.cost);

    // Print the fitted parameters
    model.print_results();

    // Compare to true values
    println!("\nComparison to true values:");
    println!("  True amplitude (a): {:.6}", true_a);
    println!("  True center (b): {:.6}", true_b);
    println!("  True width (c): {:.6}", true_c);
    println!("  True FWHM: {:.6}", 2.355 * true_c);

    Ok(())
}
