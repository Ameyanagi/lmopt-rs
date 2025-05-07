use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::{DifferentialEvolution, GlobalOptimizer, ParallelDifferentialEvolution};
use lmopt_rs::problem::Problem;
use ndarray::{array, Array1};
use std::f64::consts::PI;
use std::time::Instant;

// Rastrigin function - a challenging multimodal optimization problem
// f(x1, x2, ..., xn) = 10n + sum[ x_i^2 - 10*cos(2π*x_i) ]
struct RastriginFunction {
    dimension: usize,
}

impl RastriginFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for RastriginFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let mut sum = 10.0 * self.dimension as f64;
        for x in params.iter() {
            sum += x.powi(2) - 10.0 * (2.0 * PI * x).cos();
        }

        Ok(array![sum.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

// Ackley function - another challenging multimodal optimization problem
struct AckleyFunction {
    dimension: usize,
}

impl AckleyFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for AckleyFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let a = 20.0;
        let b = 0.2;
        let c = 2.0 * PI;

        let n = self.dimension as f64;

        // First sum term
        let sum1: f64 = params.iter().map(|&x| x.powi(2)).sum::<f64>() / n;
        let term1 = -a * (-b * sum1.sqrt()).exp();

        // Second sum term
        let sum2: f64 = params.iter().map(|&x| (c * x).cos()).sum::<f64>() / n;
        let term2 = -(sum2.exp());

        // Combine terms
        let result = term1 + term2 + a + std::f64::consts::E;

        Ok(array![result])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

fn main() {
    println!("Parallel vs. Sequential Global Optimization Comparison");
    println!("-----------------------------------------------------");

    // Try different problem dimensions to show parallel speedup
    let dimensions = [5, 10, 20];

    for &dimension in &dimensions {
        println!(
            "\nOptimizing Rastrigin function with {} dimensions",
            dimension
        );

        // Create problem and bounds
        let problem = RastriginFunction::new(dimension);
        let bounds = vec![(-5.12, 5.12); dimension];

        // Configure sequential optimizer
        let sequential_optimizer = DifferentialEvolution::new()
            .with_population_size(50)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_seed(42);

        // Run sequential optimization
        let start_time = Instant::now();
        let sequential_result = sequential_optimizer
            .optimize(&problem, &bounds, 50, 20, 1e-4)
            .unwrap();
        let sequential_time = start_time.elapsed();

        // Configure parallel optimizer
        let parallel_optimizer = ParallelDifferentialEvolution::new()
            .with_population_size(50)
            .with_crossover_probability(0.9)
            .with_differential_weight(0.8)
            .with_seed(42);

        // Run parallel optimization
        let start_time = Instant::now();
        let parallel_result = parallel_optimizer
            .optimize(&problem, &bounds, 50, 20, 1e-4)
            .unwrap();
        let parallel_time = start_time.elapsed();

        // Print results
        println!("Sequential optimization:");
        println!("  Time: {:?}", sequential_time);
        println!("  Cost: {:.6}", sequential_result.cost);
        println!("  Function evaluations: {}", sequential_result.func_evals);
        println!(
            "  Best parameters: [{:.3}, {:.3}, ...]",
            sequential_result.params[0], sequential_result.params[1]
        );

        println!("Parallel optimization:");
        println!("  Time: {:?}", parallel_time);
        println!("  Cost: {:.6}", parallel_result.cost);
        println!("  Function evaluations: {}", parallel_result.func_evals);
        println!(
            "  Best parameters: [{:.3}, {:.3}, ...]",
            parallel_result.params[0], parallel_result.params[1]
        );

        // Calculate speedup
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    }

    println!("\nOptimizing Ackley function");
    println!("-------------------------");

    // Create a problem with expensive function evaluations
    let dimension = 10;
    let problem = AckleyFunction::new(dimension);
    let bounds = vec![(-32.768, 32.768); dimension];

    // Try with local optimization enabled
    let sequential_optimizer = DifferentialEvolution::new()
        .with_population_size(40)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8)
        .with_local_optimization(true)
        .with_local_optimization_iterations(100)
        .with_seed(42);

    let start_time = Instant::now();
    let sequential_result = sequential_optimizer
        .optimize(&problem, &bounds, 30, 15, 1e-4)
        .unwrap();
    let sequential_time = start_time.elapsed();

    let parallel_optimizer = ParallelDifferentialEvolution::new()
        .with_population_size(40)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8)
        .with_local_optimization(true)
        .with_local_optimization_iterations(100)
        .with_seed(42);

    let start_time = Instant::now();
    let parallel_result = parallel_optimizer
        .optimize(&problem, &bounds, 30, 15, 1e-4)
        .unwrap();
    let parallel_time = start_time.elapsed();

    // Print results
    println!("Sequential optimization with local refinement:");
    println!("  Time: {:?}", sequential_time);
    println!("  Cost: {:.6}", sequential_result.cost);
    println!("  Function evaluations: {}", sequential_result.func_evals);

    println!("Parallel optimization with local refinement:");
    println!("  Time: {:?}", parallel_time);
    println!("  Cost: {:.6}", parallel_result.cost);
    println!("  Function evaluations: {}", parallel_result.func_evals);

    // Calculate speedup
    let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    println!("\nNote: Actual speedup depends on your CPU's core count, problem dimension,");
    println!("and the computational cost of the objective function evaluation.");
}
