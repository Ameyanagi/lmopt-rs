use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::lm::{LevenbergMarquardt, LmConfig, ParallelLevenbergMarquardt};
use lmopt_rs::problem::Problem;
use ndarray::{Array1, Array2};
use std::time::Instant;

// A simple nonlinear model for demonstration: y = a * exp(-b * x) + c
struct ExponentialModel {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    size: usize, // Number of data points
}

impl ExponentialModel {
    fn new(size: usize) -> Self {
        // Generate synthetic data
        let mut x_data = Array1::zeros(size);
        let mut y_data = Array1::zeros(size);

        // Set x values uniformly across range [0, 10]
        for i in 0..size {
            x_data[i] = 10.0 * (i as f64) / (size as f64);
        }

        // Generate y values with true parameters [a=5.0, b=0.5, c=1.0] plus noise
        let a_true = 5.0;
        let b_true = 0.5;
        let c_true = 1.0;

        for i in 0..size {
            let x = x_data[i];
            // Add random noise to make the fit more realistic
            let noise = 0.1 * (rand::random::<f64>() - 0.5);
            y_data[i] = a_true * (-b_true * x).exp() + c_true + noise;
        }

        Self {
            x_data,
            y_data,
            size,
        }
    }
}

impl Problem for ExponentialModel {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 3 {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected 3 parameters, got {}",
                params.len()
            )));
        }

        let a = params[0];
        let b = params[1];
        let c = params[2];

        let mut residuals = Array1::zeros(self.size);

        for i in 0..self.size {
            let x = self.x_data[i];
            let y = self.y_data[i];
            let model_y = a * (-b * x).exp() + c;
            residuals[i] = model_y - y;
        }

        Ok(residuals)
    }

    fn parameter_count(&self) -> usize {
        3 // a, b, c
    }

    fn residual_count(&self) -> usize {
        self.size
    }

    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        let a = params[0];
        let b = params[1];
        let c = params[2];

        let mut jac = Array2::zeros((self.size, 3));

        for i in 0..self.size {
            let x = self.x_data[i];
            let exp_term = (-b * x).exp();

            // ∂f/∂a = exp(-b*x)
            jac[[i, 0]] = exp_term;

            // ∂f/∂b = -a * x * exp(-b*x)
            jac[[i, 1]] = -a * x * exp_term;

            // ∂f/∂c = 1
            jac[[i, 2]] = 1.0;
        }

        Ok(jac)
    }

    fn has_custom_jacobian(&self) -> bool {
        true
    }
}

fn main() {
    println!("Parallel vs. Sequential Levenberg-Marquardt Comparison");
    println!("-----------------------------------------------------");

    // Try different problem sizes to demonstrate parallel speedup
    let sizes = [100, 1000, 10000, 100000];

    for &size in &sizes {
        println!("\nProblem size: {} data points", size);

        let model = ExponentialModel::new(size);
        let initial_params = Array1::from_vec(vec![1.0, 0.1, 0.5]); // Starting far from true values

        // Sequential LM
        let lm_sequential = LevenbergMarquardt::with_default_config();
        let start_time = Instant::now();
        let result_sequential = lm_sequential
            .minimize(&model, initial_params.clone())
            .unwrap();
        let sequential_duration = start_time.elapsed();

        // Parallel LM
        let lm_parallel = ParallelLevenbergMarquardt::with_default_config();
        let start_time = Instant::now();
        let result_parallel = lm_parallel
            .minimize(&model, initial_params.clone())
            .unwrap();
        let parallel_duration = start_time.elapsed();

        // Print results
        println!("Sequential LM: {:?}", sequential_duration);
        println!(
            "Parameters: a={:.3}, b={:.3}, c={:.3}",
            result_sequential.params[0], result_sequential.params[1], result_sequential.params[2]
        );
        println!(
            "Cost: {:.6e}, Iterations: {}",
            result_sequential.cost, result_sequential.iterations
        );

        println!("Parallel LM:   {:?}", parallel_duration);
        println!(
            "Parameters: a={:.3}, b={:.3}, c={:.3}",
            result_parallel.params[0], result_parallel.params[1], result_parallel.params[2]
        );
        println!(
            "Cost: {:.6e}, Iterations: {}",
            result_parallel.cost, result_parallel.iterations
        );

        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    }

    // Demonstrate automatic Jacobian calculation using parallel processing
    println!("\nDemonstration with automatic Jacobian calculation");

    // Define a problem without a custom Jacobian implementation
    struct SimpleModel {
        model: ExponentialModel,
    }

    impl SimpleModel {
        fn new(size: usize) -> Self {
            Self {
                model: ExponentialModel::new(size),
            }
        }
    }

    impl Problem for SimpleModel {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            self.model.eval(params)
        }

        fn parameter_count(&self) -> usize {
            self.model.parameter_count()
        }

        fn residual_count(&self) -> usize {
            self.model.residual_count()
        }

        // No custom Jacobian implementation - will use numerical differentiation
    }

    let size = 10000;
    println!("\nProblem size: {} data points, automatic Jacobian", size);

    let model = SimpleModel::new(size);
    let initial_params = Array1::from_vec(vec![1.0, 0.1, 0.5]);

    // Sequential LM with automatic Jacobian
    let lm_sequential = LevenbergMarquardt::with_default_config();
    let start_time = Instant::now();
    let result_sequential = lm_sequential
        .minimize(&model, initial_params.clone())
        .unwrap();
    let sequential_duration = start_time.elapsed();

    // Parallel LM with automatic Jacobian
    let lm_parallel = ParallelLevenbergMarquardt::with_default_config();
    let start_time = Instant::now();
    let result_parallel = lm_parallel
        .minimize(&model, initial_params.clone())
        .unwrap();
    let parallel_duration = start_time.elapsed();

    // Print results
    println!("Sequential LM: {:?}", sequential_duration);
    println!(
        "Cost: {:.6e}, Iterations: {}",
        result_sequential.cost, result_sequential.iterations
    );

    println!("Parallel LM:   {:?}", parallel_duration);
    println!(
        "Cost: {:.6e}, Iterations: {}",
        result_parallel.cost, result_parallel.iterations
    );

    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    println!("\nNote: Actual speedup depends on your CPU's core count and the problem size.");
}
