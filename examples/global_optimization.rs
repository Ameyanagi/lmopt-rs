//! Example of using global optimization methods.
//!
//! This example demonstrates how to use various global optimization methods
//! to find the global minimum of a function with multiple local minima.

use lmopt_rs::{
    optimize_global, optimize_global_param_problem, parameters::Parameters, BasinHopping,
    DifferentialEvolution, GlobalOptResult, GlobalOptimizer, HybridGlobal, ParameterProblem,
    Problem, SimulatedAnnealing,
};

use ndarray::{array, Array1, Array2};
use std::error::Error;
use std::time::Instant;

// Define a test problem with multiple local minima
struct MultiWellProblem;

impl Problem for MultiWellProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Check parameter dimensions
        if params.len() != 2 {
            return Err(lmopt_rs::LmOptError::DimensionMismatch(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        let x = params[0];
        let y = params[1];

        // Rastrigin function: has multiple local minima
        // f(x, y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))
        // Global minimum at (0, 0) with value 0
        let term1 = 20.0 + x.powi(2) + y.powi(2);
        let term2 = 10.0
            * ((2.0 * std::f64::consts::PI * x).cos() + (2.0 * std::f64::consts::PI * y).cos());
        let value = term1 - term2;

        // Return as a single residual
        Ok(array![value.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residual_count(&self) -> usize {
        1
    }
}

// Define a parameter problem version
struct MultiWellParameterProblem {
    params: Parameters,
}

impl MultiWellParameterProblem {
    fn new() -> Self {
        let mut params = Parameters::new();
        params.add_param_with_bounds("x", 2.0, -5.0, 5.0).unwrap();
        params.add_param_with_bounds("y", 2.0, -5.0, 5.0).unwrap();

        Self { params }
    }
}

impl ParameterProblem for MultiWellParameterProblem {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.params
    }

    fn parameters(&self) -> &Parameters {
        &self.params
    }

    fn eval_with_parameters(&self) -> lmopt_rs::Result<Array1<f64>> {
        // Get parameter values
        let x = self.params.get("x").unwrap().value();
        let y = self.params.get("y").unwrap().value();

        // Calculate Rastrigin function
        let term1 = 20.0 + x.powi(2) + y.powi(2);
        let term2 = 10.0
            * ((2.0 * std::f64::consts::PI * x).cos() + (2.0 * std::f64::consts::PI * y).cos());
        let value = term1 - term2;

        // Return as a single residual
        Ok(array![value.sqrt()])
    }

    fn residual_count(&self) -> usize {
        1
    }
}

// Define a more challenging 10-dimensional problem
struct HighDimensionalProblem;

impl Problem for HighDimensionalProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Check parameter dimensions
        if params.len() != 10 {
            return Err(lmopt_rs::LmOptError::DimensionMismatch(format!(
                "Expected 10 parameters, got {}",
                params.len()
            )));
        }

        // 10-dimensional Rosenbrock function
        // f(x) = sum_{i=0}^{n-2} [ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
        // Global minimum at (1, 1, ..., 1) with value 0
        let mut value = 0.0;
        for i in 0..9 {
            let term1 = 100.0 * (params[i + 1] - params[i].powi(2)).powi(2);
            let term2 = (1.0 - params[i]).powi(2);
            value += term1 + term2;
        }

        // Return as a single residual
        Ok(array![value.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        10
    }

    fn residual_count(&self) -> usize {
        1
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Global Optimization Example");
    println!("===========================\n");

    // Test different global optimization methods on the 2D problem
    println!("2D Rastrigin Function");
    println!("-------------------");
    println!("This function has multiple local minima with a global minimum at (0, 0).\n");

    // Define bounds
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

    // 1. Simulated Annealing
    println!("1. Simulated Annealing");
    let problem = MultiWellProblem;
    let optimizer = SimulatedAnnealing::new();

    let start = Instant::now();
    let result = optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!(
        "  Solution: ({:.6}, {:.6})",
        result.params[0], result.params[1]
    );
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // 2. Differential Evolution
    println!("2. Differential Evolution");
    let optimizer = DifferentialEvolution::new();

    let start = Instant::now();
    let result = optimizer.optimize(&problem, &bounds, 100, 20, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!(
        "  Solution: ({:.6}, {:.6})",
        result.params[0], result.params[1]
    );
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // 3. Basin Hopping
    println!("3. Basin Hopping");
    let optimizer = BasinHopping::new();

    let start = Instant::now();
    let result = optimizer.optimize(&problem, &bounds, 20, 5, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!(
        "  Solution: ({:.6}, {:.6})",
        result.params[0], result.params[1]
    );
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // 4. Hybrid Global
    println!("4. Hybrid Global (Differential Evolution + LM)");
    let optimizer = HybridGlobal::new();

    let start = Instant::now();
    let result = optimizer.optimize(&problem, &bounds, 50, 10, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!(
        "  Solution: ({:.6}, {:.6})",
        result.params[0], result.params[1]
    );
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // 5. Using with parameter problem
    println!("5. Using with parameter problem");
    let mut param_problem = MultiWellParameterProblem::new();

    let start = Instant::now();
    let result = optimize_global_param_problem(&mut param_problem, 50, 10, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!(
        "  Solution: x={:.6}, y={:.6}",
        param_problem.parameters().get("x").unwrap().value(),
        param_problem.parameters().get("y").unwrap().value()
    );
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // 6. Higher-dimensional problem
    println!("6. 10-dimensional Rosenbrock Function");
    println!("----------------------------------");
    println!("This is a challenging function that tests the optimizer's ability");
    println!("to handle higher-dimensional parameter spaces.\n");

    let high_dim_problem = HighDimensionalProblem;

    // Define bounds for all 10 parameters
    let high_dim_bounds = vec![(-5.0, 5.0); 10];

    // Use differential evolution for this problem
    let optimizer = DifferentialEvolution::new();

    let start = Instant::now();
    let result = optimizer.optimize(&high_dim_problem, &high_dim_bounds, 200, 50, 1e-6)?;
    let duration = start.elapsed();

    println!("  Success: {}", result.success);
    println!("  Cost: {:.6e}", result.cost);
    println!("  Solution:");
    for (i, &value) in result.params.iter().enumerate() {
        println!("    x{} = {:.6}", i, value);
    }
    println!("  Function evaluations: {}", result.func_evals);
    println!("  Elapsed time: {:.2?}", duration);
    println!();

    // Performance comparison on the 2D problem
    println!("Performance Comparison (10 runs each)");
    println!("---------------------------------");

    let methods = [
        "Simulated Annealing",
        "Differential Evolution",
        "Basin Hopping",
        "Hybrid Global",
    ];

    for &method in &methods {
        let mut total_evals = 0;
        let mut total_time = std::time::Duration::new(0, 0);
        let mut success_count = 0;
        let mut best_cost = std::f64::INFINITY;

        println!("Method: {}", method);

        for run in 0..10 {
            let start = Instant::now();

            let result = match method {
                "Simulated Annealing" => {
                    let optimizer = SimulatedAnnealing::new();
                    optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6)?
                }
                "Differential Evolution" => {
                    let optimizer = DifferentialEvolution::new();
                    optimizer.optimize(&problem, &bounds, 100, 20, 1e-6)?
                }
                "Basin Hopping" => {
                    let optimizer = BasinHopping::new();
                    optimizer.optimize(&problem, &bounds, 20, 5, 1e-6)?
                }
                "Hybrid Global" => {
                    let optimizer = HybridGlobal::new();
                    optimizer.optimize(&problem, &bounds, 50, 10, 1e-6)?
                }
                _ => unreachable!(),
            };

            let duration = start.elapsed();

            if result.success {
                success_count += 1;
            }

            total_evals += result.func_evals;
            total_time += duration;

            if result.cost < best_cost {
                best_cost = result.cost;
            }

            print!(".");
            if (run + 1) % 10 == 0 {
                println!();
            }
        }

        println!("  Success rate: {}/10", success_count);
        println!("  Average evaluations: {:.1}", total_evals as f64 / 10.0);
        println!("  Average time: {:.2?}", total_time / 10);
        println!("  Best cost: {:.6e}", best_cost);
        println!();
    }

    println!("Example completed successfully!");
    Ok(())
}
