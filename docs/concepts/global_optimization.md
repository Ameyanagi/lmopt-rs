# Global Optimization

Nonlinear least-squares problems often have multiple local minima, making it challenging to find the global optimum. The `lmopt-rs` library provides several global optimization methods to address this challenge.

## Related Documentation
- [Getting Started Guide](../getting_started.md)
- [Parameter System](./parameters.md) - Parameter management for optimization
- [Model System](./models.md) - Models that can be globally optimized
- [Levenberg-Marquardt Algorithm](./lm_algorithm.md) - Local optimization used with global methods
- [Global Optimization Examples](../examples/global_optimization.md)
- [Comparison with Other Libraries](../comparison.md)

## Core Global Optimization Concepts

The global optimization methods in `lmopt-rs` use different strategies to search the parameter space and locate the global minimum:

1. **Simulated Annealing**: Probabilistic technique inspired by metallurgical annealing
2. **Differential Evolution**: Population-based evolutionary algorithm
3. **Basin Hopping**: Iterative local optimization with random perturbations
4. **Hybrid Methods**: Combination of global and local optimization

## Using Global Optimization

```rust
use lmopt_rs::{
    optimize_global, Problem, DifferentialEvolution, 
    GlobalOptimizer, GlobalOptResult
};
use ndarray::Array1;

// Define your problem
struct MultiMinimaProblem;

impl Problem for MultiMinimaProblem {
    // Implement the Problem trait...
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the problem
    let problem = MultiMinimaProblem;
    
    // Define parameter bounds (min, max)
    let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];
    
    // Choose a global optimization method
    let optimizer = DifferentialEvolution::new();
    
    // Run the optimization
    let result = optimizer.optimize(
        &problem,    // The problem to solve
        &bounds,     // Parameter bounds
        100,         // Number of iterations
        20,          // Population size
        1e-6         // Tolerance
    )?;
    
    // Print results
    println!("Optimization successful: {}", result.success);
    println!("Final parameters: {:?}", result.params);
    println!("Final cost: {}", result.cost);
    println!("Function evaluations: {}", result.func_evals);
    
    Ok(())
}
```

## Available Methods

### Simulated Annealing

Simulated annealing mimics the physical process of heating a material and then slowly cooling it to decrease defects:

```rust
use lmopt_rs::SimulatedAnnealing;

// Create a simulated annealing optimizer
let optimizer = SimulatedAnnealing::new()
    .with_initial_temp(10.0)   // Starting temperature
    .with_cooling_factor(0.95); // How quickly temperature decreases

// Run the optimization
let result = optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6)?;
```

Key parameters:
- **Iterations**: Number of cooling steps (outer loop)
- **Steps**: Number of steps at each temperature (inner loop)
- **Initial temperature**: Controls initial acceptance probability
- **Cooling factor**: Rate at which temperature decreases

### Differential Evolution

A population-based evolutionary algorithm that evolves a population of candidate solutions:

```rust
use lmopt_rs::DifferentialEvolution;

// Create a differential evolution optimizer
let optimizer = DifferentialEvolution::new()
    .with_crossover_probability(0.7)
    .with_differential_weight(0.8);

// Run the optimization
let result = optimizer.optimize(&problem, &bounds, 100, 20, 1e-6)?;
```

Key parameters:
- **Iterations**: Number of generations
- **Population size**: Number of candidate solutions
- **Crossover probability**: Controls mixing of parent and child vectors
- **Differential weight**: Scales the influence of difference vectors

### Basin Hopping

Combines local optimization with random perturbations:

```rust
use lmopt_rs::BasinHopping;

// Create a basin hopping optimizer
let optimizer = BasinHopping::new()
    .with_step_size(1.0)          // Size of random perturbations
    .with_temperature(1.0)        // Controls acceptance probability
    .with_local_iter(10);         // Local optimization iterations

// Run the optimization
let result = optimizer.optimize(&problem, &bounds, 20, 5, 1e-6)?;
```

Key parameters:
- **Iterations**: Number of basin hops
- **Steps**: Number of random perturbations at each iteration
- **Step size**: Controls magnitude of parameter perturbations
- **Temperature**: Affects acceptance probability of uphill moves

### Hybrid Global

Combines global search with local refinement:

```rust
use lmopt_rs::HybridGlobal;

// Create a hybrid optimizer
let optimizer = HybridGlobal::new()
    .with_global_method(GlobalMethod::DifferentialEvolution)
    .with_local_method(LocalMethod::LevenbergMarquardt);

// Run the optimization
let result = optimizer.optimize(&problem, &bounds, 50, 10, 1e-6)?;
```

This method first performs a global search to locate promising regions, then refines the solution with local optimization.

## Using with Parameter Problems

For problems that use the parameter system:

```rust
use lmopt_rs::{optimize_global_param_problem, ParameterProblem};

// Define a problem that implements ParameterProblem
struct ParamProblem {
    params: Parameters,
    // other fields...
}

impl ParameterProblem for ParamProblem {
    // Implement the trait...
}

// Create the problem with bounded parameters
let mut problem = ParamProblem::new();
problem.parameters_mut().add_param_with_bounds("x", 0.0, -10.0, 10.0)?;
problem.parameters_mut().add_param_with_bounds("y", 0.0, -10.0, 10.0)?;

// Run global optimization
let result = optimize_global_param_problem(&mut problem, 100, 20, 1e-6)?;

// Access optimized parameters
println!("Optimized x: {}", problem.parameters().get("x").unwrap().value());
println!("Optimized y: {}", problem.parameters().get("y").unwrap().value());
```

The `optimize_global_param_problem` function automatically extracts bounds from the parameters and selects an appropriate global optimization method.

## Custom Global Optimizers

You can create custom global optimization methods by implementing the `GlobalOptimizer` trait:

```rust
use lmopt_rs::{GlobalOptimizer, GlobalOptResult, Problem};

struct MyCustomOptimizer {
    // Custom configuration...
}

impl GlobalOptimizer for MyCustomOptimizer {
    fn optimize<P: Problem>(
        &self,
        problem: &P,
        bounds: &[(f64, f64)],
        iterations: usize,
        population_size: usize,
        tolerance: f64
    ) -> lmopt_rs::Result<GlobalOptResult> {
        // Custom optimization algorithm...
    }
}
```

## Advanced Topics

### Multi-Start Optimization

A simple yet effective approach is to run multiple local optimizations from different starting points:

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::Array1;
use rand::Rng;

fn multi_start_optimization<P: Problem>(
    problem: &P,
    bounds: &[(f64, f64)],
    n_starts: usize
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let param_count = problem.parameter_count();
    let mut best_cost = f64::INFINITY;
    let mut best_params = Array1::zeros(param_count);
    let mut rng = rand::thread_rng();
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    for _ in 0..n_starts {
        // Generate random starting point within bounds
        let mut initial_params = Array1::zeros(param_count);
        for i in 0..param_count {
            let (lower, upper) = bounds[i];
            initial_params[i] = rng.gen_range(lower..upper);
        }
        
        // Run local optimization
        let result = optimizer.minimize(problem, initial_params)?;
        
        // Update best solution if better
        if result.success && result.cost < best_cost {
            best_cost = result.cost;
            best_params = result.params.clone();
        }
    }
    
    Ok(best_params)
}
```

### Parallel Global Optimization

Utilize multiple CPU cores for faster optimization:

```rust
use rayon::prelude::*;

// Parallel multi-start optimization
fn parallel_multi_start<P: Problem + Sync>(
    problem: &P,
    bounds: &[(f64, f64)],
    n_starts: usize
) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
    let param_count = problem.parameter_count();
    
    // Generate starting points
    let starting_points: Vec<_> = (0..n_starts)
        .map(|_| {
            let mut params = Array1::zeros(param_count);
            let mut rng = rand::thread_rng();
            for i in 0..param_count {
                let (lower, upper) = bounds[i];
                params[i] = rng.gen_range(lower..upper);
            }
            params
        })
        .collect();
    
    // Run optimizations in parallel
    let results: Vec<_> = starting_points.par_iter()
        .map(|start| {
            let mut optimizer = LevenbergMarquardt::with_default_config();
            optimizer.minimize(problem, start.clone())
        })
        .collect::<Result<Vec<_>, _>>()?;
    
    // Find best result
    let best = results.iter()
        .filter(|r| r.success)
        .min_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap())
        .ok_or("No successful optimization")?;
    
    Ok(best.params.clone())
}
```

### Adaptive Parameter Control

For differential evolution, adaptive parameter control can improve performance:

```rust
// Example of adaptive differential evolution
let optimizer = DifferentialEvolution::new()
    .with_adaptive_params(true)  // Enable adaptive parameters
    .with_adaptation_rate(0.1);  // Control adaptation speed
```

### Hybrid Strategy Selection

Automatically select the best global optimization strategy:

```rust
use lmopt_rs::global_opt::auto_select_optimizer;

// Automatically select the best optimizer for the problem
let optimizer = auto_select_optimizer(problem, &bounds)?;
let result = optimizer.optimize(problem, &bounds, 100, 20, 1e-6)?;
```

## Best Practices

1. **Define tight parameter bounds**: Narrower bounds significantly improve performance.

2. **Try multiple methods**: Different methods work better for different problems.

3. **Adjust control parameters**: Default parameters may not be optimal for your specific problem.

4. **Use sufficient iterations**: Global optimization generally requires more function evaluations than local methods.

5. **Consider problem structure**: If the problem allows, decompose it into smaller subproblems.

6. **Verify global minimum**: Run multiple optimizations to ensure you've found the global minimum.

7. **Use hybrid approaches**: Often the most effective strategy is to combine global exploration with local refinement.