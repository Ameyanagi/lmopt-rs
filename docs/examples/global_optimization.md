# Global Optimization Examples

This guide demonstrates how to use global optimization methods in `lmopt-rs` to find the global minimum of functions with multiple local minima.

## Related Documentation
- [Getting Started Guide](../getting_started.md)
- [Global Optimization Concepts](../concepts/global_optimization.md)
- [Parameter System](../concepts/parameters.md)
- [Levenberg-Marquardt Algorithm](../concepts/lm_algorithm.md) - Local optimization component
- [Basic Fitting](./basic_fitting.md) - Start here for simpler optimization
- [Composite Models](./composite_models.md) - Complex models that may require global optimization
- [Comparison with Other Libraries](../comparison.md)

## Introduction to Global Optimization

When your optimization problem has multiple local minima, standard local optimization methods like Levenberg-Marquardt may get stuck in a non-optimal solution. Global optimization algorithms are designed to explore the parameter space more thoroughly to find the best possible solution.

`lmopt-rs` provides several global optimization algorithms:

1. **Simulated Annealing**: Inspired by the annealing process in metallurgy
2. **Differential Evolution**: A population-based evolutionary algorithm
3. **Basin Hopping**: Combines random jumps with local optimization
4. **Hybrid Global Optimization**: A meta-algorithm that combines multiple approaches

## Basic Usage

The simplest way to use global optimization is with the `optimize_global` function:

```rust
use lmopt_rs::{optimize_global, Problem};
use ndarray::{array, Array1};

// Define a problem with multiple local minima
struct MultiWellProblem;

impl Problem for MultiWellProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        let x = params[0];
        let y = params[1];
        
        // Rastrigin function: has multiple local minima
        // f(x, y) = 20 + x^2 + y^2 - 10(cos(2πx) + cos(2πy))
        let term1 = 20.0 + x.powi(2) + y.powi(2);
        let term2 = 10.0 * ((2.0 * std::f64::consts::PI * x).cos() + 
                          (2.0 * std::f64::consts::PI * y).cos());
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

// Create the problem
let problem = MultiWellProblem;

// Define bounds for the parameters
let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

// Run global optimization
let result = optimize_global(
    &problem,    // Problem to solve
    &bounds,     // Parameter bounds
    100,         // Number of iterations
    20,          // Population size
    1e-6         // Tolerance
).unwrap();

println!("Global minimum found at: ({:.3}, {:.3})", 
         result.params[0], result.params[1]);
println!("Function value: {:.6}", result.cost);
```

## Using Specific Algorithms

Each global optimization algorithm has strengths and weaknesses for different types of problems. You can choose a specific algorithm based on your problem characteristics.

### Simulated Annealing

Simulated Annealing is inspired by the annealing process in metallurgy. It allows moves to worse solutions with a probability that decreases over time, helping escape local minima.

```rust
use lmopt_rs::{GlobalOptimizer, SimulatedAnnealing, Problem};

// Create and configure the optimizer
let optimizer = SimulatedAnnealing::new()
    .with_initial_temp(10.0)        // Starting temperature
    .with_cooling_factor(0.95)      // How quickly temperature decreases
    .with_step_size(1.0);           // Size of random moves

// Run the optimization
let result = optimizer.optimize(
    &problem,
    &bounds,
    1000,   // Maximum iterations
    100,    // Maximum iterations without improvement
    1e-6    // Tolerance
).unwrap();
```

### Differential Evolution

Differential Evolution is a population-based evolutionary algorithm that maintains a population of candidate solutions and evolves them through mutation, crossover, and selection.

```rust
use lmopt_rs::{GlobalOptimizer, DifferentialEvolution, Problem};

// Create and configure the optimizer
let optimizer = DifferentialEvolution::new()
    .with_differential_weight(0.8)      // Mutation factor F
    .with_crossover_probability(0.7);   // Crossover probability CR

// Run the optimization
let result = optimizer.optimize(
    &problem,
    &bounds,
    100,    // Maximum iterations
    30,     // Population size (also used as maximum no improvement)
    1e-6    // Tolerance
).unwrap();
```

### Basin Hopping

Basin Hopping combines random jumps with local optimization, making it effective for problems with many local minima separated by barriers.

```rust
use lmopt_rs::{GlobalOptimizer, BasinHopping, Problem};

// Create and configure the optimizer
let optimizer = BasinHopping::new()
    .with_step_size(1.0)            // Size of random jumps
    .with_temperature(1.0)          // Temperature for acceptance probability
    .with_local_iterations(10);     // Local optimization iterations per jump

// Run the optimization
let result = optimizer.optimize(
    &problem,
    &bounds,
    50,     // Maximum iterations (jumps)
    10,     // Maximum iterations without improvement
    1e-6    // Tolerance
).unwrap();
```

### Hybrid Global Optimization

The Hybrid Global approach combines multiple methods to leverage their strengths and increase the chance of finding the global minimum.

```rust
use lmopt_rs::{GlobalOptimizer, HybridGlobal, Problem};

// Create the hybrid optimizer
let optimizer = HybridGlobal::new()
    .with_global_iterations(50)     // Number of global exploration iterations
    .with_population_size(20)       // Population size for evolutionary methods
    .with_local_iterations(100);    // Maximum iterations for local refinement

// Run the optimization
let result = optimizer.optimize(
    &problem,
    &bounds,
    50,     // Maximum iterations
    20,     // Population size (also used as maximum no improvement)
    1e-6    // Tolerance
).unwrap();
```

## Working with Parameter Problems

You can use global optimization with the parameter system using `optimize_global_param_problem`:

```rust
use lmopt_rs::{optimize_global_param_problem, parameters::Parameters, ParameterProblem};
use ndarray::Array1;

// Create a parameter problem implementation
struct MyParameterProblem {
    parameters: Parameters,
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl ParameterProblem for MyParameterProblem {
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }
    
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }
    
    fn eval_with_parameters(&self) -> lmopt_rs::Result<Array1<f64>> {
        // Calculate residuals using current parameters
        // ...
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

// Create problem instance
let mut problem = MyParameterProblem {
    parameters: Parameters::new(),
    x_data: /* ... */,
    y_data: /* ... */,
};

// Add parameters with bounds
problem.parameters_mut().add_param_with_bounds("a", 1.0, -10.0, 10.0).unwrap();
problem.parameters_mut().add_param_with_bounds("b", 1.0, -10.0, 10.0).unwrap();

// Run global optimization
let result = optimize_global_param_problem(
    &mut problem,
    100,    // Maximum iterations
    20,     // Population size
    1e-6    // Tolerance
).unwrap();

// Access optimized parameters
let a_opt = problem.parameters().get("a").unwrap().value();
let b_opt = problem.parameters().get("b").unwrap().value();
```

## Example: Fitting a Multi-Peak Spectrum

This example shows how to use global optimization to fit a spectrum with multiple peaks, where local optimization might struggle:

```rust
use lmopt_rs::{optimize_global_param_problem, GlobalOptimizer, BasinHopping};
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, add};
use ndarray::Array1;

// Generate synthetic data with multiple peaks
let x = Array1::linspace(0.0, 10.0, 200);
let mut y = Array1::zeros(200);

// Add three Gaussian peaks and noise
for i in 0..200 {
    let x_val = x[i];
    
    // First peak
    let g1 = 3.0 * (-(x_val - 2.0).powi(2) / (2.0 * 0.5f64.powi(2))).exp();
    
    // Second peak
    let g2 = 4.0 * (-(x_val - 5.0).powi(2) / (2.0 * 0.8f64.powi(2))).exp();
    
    // Third peak
    let g3 = 2.5 * (-(x_val - 8.0).powi(2) / (2.0 * 0.6f64.powi(2))).exp();
    
    // Add noise
    let noise = rand::thread_rng().gen_range(-0.2..0.2);
    y[i] = g1 + g2 + g3 + 0.5 + noise;
}

// Create a composite model with three Gaussian peaks
let mut g1 = GaussianModel::new("g1_", false);
let mut g2 = GaussianModel::new("g2_", false);
let mut g3 = GaussianModel::new("g3_", false);
let mut model = add(g1, g2);
model = add(model, g3);
model.add_baseline(0.5);

// Set initial guesses intentionally off to demonstrate global optimization
g1.parameters_mut().get_mut("g1_center").unwrap().set_value(1.0).unwrap();
g2.parameters_mut().get_mut("g2_center").unwrap().set_value(4.0).unwrap();
g3.parameters_mut().get_mut("g3_center").unwrap().set_value(7.0).unwrap();

// Set data for the model
model.set_data(x.clone(), y.clone());

// First, try standard local optimization
let local_result = fit(&mut model, x.clone(), y.clone()).unwrap();
println!("Local optimization result:");
println!("Success: {}", local_result.success);
println!("Cost: {:.6}", local_result.cost);

// Now try global optimization
let optimizer = BasinHopping::new()
    .with_step_size(0.5)
    .with_temperature(1.0);

let global_result = optimizer.run_optimization(
    &mut model,
    30,     // Maximum iterations
    10,     // Maximum iterations without improvement
    1e-6    // Tolerance
).unwrap();

println!("\nGlobal optimization result:");
println!("Success: {}", global_result.success);
println!("Cost: {:.6}", global_result.cost);

// Compare the peak positions
println!("\nPeak centers from local optimization:");
println!("g1_center: {:.3}", model.parameters().get("g1_center").unwrap().value());
println!("g2_center: {:.3}", model.parameters().get("g2_center").unwrap().value());
println!("g3_center: {:.3}", model.parameters().get("g3_center").unwrap().value());
```

## Choosing the Right Algorithm

Each global optimization algorithm has different strengths:

| Algorithm | Best For | When to Avoid |
|-----------|----------|---------------|
| **Simulated Annealing** | - Problems with few parameters<br>- Continuous parameter spaces<br>- When you need a simple approach | - High-dimensional problems<br>- When very precise solutions are needed |
| **Differential Evolution** | - Medium to high dimensions<br>- Problems with complex landscapes<br>- When parallelization is possible | - Very expensive function evaluations<br>- When you need deterministic results |
| **Basin Hopping** | - Problems with distinct basins<br>- When local optimization is efficient<br>- Problems with many local minima | - Very flat landscapes<br>- Problems with many constraints |
| **Hybrid Global** | - Complex real-world problems<br>- When you're not sure which method is best<br>- Problems that need both exploration and precision | - Simple problems where a specific method is known to work well<br>- Very time-constrained applications |

## Performance Tips

1. **Set appropriate bounds**: Tighter bounds can significantly improve performance
2. **Normalize parameters**: Try to keep parameters on similar scales
3. **Use problem-specific knowledge**: Incorporate domain knowledge into initial guesses
4. **Adjust algorithm parameters**: Tune step size, temperature, etc. based on your specific problem
5. **Consider hybrid approaches**: Different algorithms may work better for different phases of optimization

## Common Issues and Solutions

| Issue | Possible Solutions |
|-------|-------------------|
| **Algorithm gets stuck** | - Increase temperature/step size<br>- Try a different algorithm<br>- Use broader bounds |
| **Too slow convergence** | - Decrease population size<br>- Use tighter bounds<br>- Improve initial guesses |
| **Poor solution quality** | - Increase iterations/population size<br>- Try a different algorithm<br>- Combine with local optimization |
| **Parameter values at bounds** | - Extend bounds<br>- Reconsider the model<br>- Check for parameter correlations |

## Extended Example: Using Multiple Algorithms

For challenging problems, you can try multiple algorithms and compare results:

```rust
use lmopt_rs::{GlobalOptimizer, SimulatedAnnealing, DifferentialEvolution, BasinHopping, HybridGlobal};

// Define problem, bounds, etc. as before

// Try different global optimization methods
let algorithms = [
    ("Simulated Annealing", SimulatedAnnealing::new().optimize(&problem, &bounds, 1000, 100, 1e-6)),
    ("Differential Evolution", DifferentialEvolution::new().optimize(&problem, &bounds, 100, 30, 1e-6)),
    ("Basin Hopping", BasinHopping::new().optimize(&problem, &bounds, 50, 10, 1e-6)),
    ("Hybrid Global", HybridGlobal::new().optimize(&problem, &bounds, 50, 20, 1e-6))
];

// Print results
println!("Algorithm Comparison:");
println!("{:<25} {:<15} {:<15} {:<15}", "Algorithm", "Best Cost", "Iterations", "Function Evals");
println!("{:-<70}", "");

for (name, result) in &algorithms {
    match result {
        Ok(res) => println!("{:<25} {:<15.6} {:<15} {:<15}", 
                           name, res.cost, res.iterations, res.func_evals),
        Err(e) => println!("{:<25} Failed: {}", name, e),
    }
}

// Find the best algorithm
let best = algorithms.iter()
    .filter_map(|(name, res)| res.as_ref().ok().map(|r| (name, r.cost)))
    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

if let Some((name, cost)) = best {
    println!("\nBest algorithm: {} with cost {:.6}", name, cost);
}
```

## Advanced: Custom Global Optimization

You can implement your own global optimization algorithm by implementing the `GlobalOptimizer` trait:

```rust
use lmopt_rs::{GlobalOptimizer, GlobalOptResult, Problem, Result};
use ndarray::Array1;
use rand::Rng;

struct MyCustomOptimizer {
    // Algorithm-specific parameters
}

impl MyCustomOptimizer {
    pub fn new() -> Self {
        Self {}
    }
}

impl GlobalOptimizer for MyCustomOptimizer {
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        max_no_improvement: usize,
        tol: f64,
    ) -> Result<GlobalOptResult> {
        // Implement your optimization algorithm here
        // ...
        
        // Return the result
        Ok(GlobalOptResult {
            params: best_params,
            cost: best_cost,
            iterations: iterations,
            func_evals: func_evals,
            success: true,
            message: "Optimization successful".to_string(),
            local_result: None,
        })
    }
}
```

This allows you to integrate custom optimization algorithms with the rest of the `lmopt-rs` ecosystem.