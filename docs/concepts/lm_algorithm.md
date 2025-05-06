# Levenberg-Marquardt Algorithm

The Levenberg-Marquardt (LM) algorithm is a powerful method for solving nonlinear least-squares problems. The `lmopt-rs` library provides a robust implementation that combines the efficiency of the algorithm with flexible parameter handling and uncertainty quantification.

## Core Algorithm Concepts

The Levenberg-Marquardt algorithm iteratively minimizes the sum of squared residuals between a model and a set of data points by updating the model parameters. It combines the advantages of gradient descent and the Gauss-Newton method, offering excellent performance for a wide range of nonlinear problems.

### Key Features

- Adaptive damping parameter (λ) that switches between gradient descent and Gauss-Newton approaches
- Efficient convergence for both well-conditioned and ill-conditioned problems
- Support for bounds constraints via parameter transformations
- Robust convergence criteria

## Mathematical Background

### The Least Squares Problem

Given a model function $f(x, \boldsymbol{\beta})$ and data points $(x_i, y_i)$, the goal is to find parameters $\boldsymbol{\beta}$ that minimize:

$$S(\boldsymbol{\beta}) = \sum_{i=1}^{m} [y_i - f(x_i, \boldsymbol{\beta})]^2 = \sum_{i=1}^{m} r_i(\boldsymbol{\beta})^2$$

where $r_i(\boldsymbol{\beta}) = y_i - f(x_i, \boldsymbol{\beta})$ are the residuals.

### The LM Update Step

The core of the algorithm is the parameter update equation:

$$(J^T J + \lambda \text{diag}(J^T J)) \Delta\boldsymbol{\beta} = J^T \mathbf{r}$$

where:
- $J$ is the Jacobian matrix (derivatives of residuals with respect to parameters)
- $\lambda$ is the damping parameter (adjusted dynamically)
- $\Delta\boldsymbol{\beta}$ is the parameter update
- $\mathbf{r}$ is the vector of residuals

## Using the Algorithm

The `lmopt-rs` library provides the `LevenbergMarquardt` struct to solve optimization problems:

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::{Array1, array};

// Create the optimizer
let mut optimizer = LevenbergMarquardt::with_default_config();

// Configure specific options
let optimizer = LevenbergMarquardt::new()
    .with_max_iterations(100)    // Maximum iterations
    .with_ftol(1e-8)             // Function tolerance
    .with_xtol(1e-8)             // Parameter tolerance
    .with_lambda(0.01)           // Initial damping parameter
    .with_lambda_up_factor(10.0) // Factor to increase lambda
    .with_lambda_down_factor(0.1); // Factor to decrease lambda

// Solve the problem
let result = optimizer.minimize(&problem, initial_params)?;

// Access the results
println!("Optimized parameters: {:?}", result.params);
println!("Final cost: {}", result.cost);
println!("Iterations: {}", result.iterations);
println!("Success: {}", result.success);
```

## Problem Definition

To use the Levenberg-Marquardt algorithm, you need to define a problem that implements the `Problem` trait:

```rust
use lmopt_rs::{Problem, Result};
use ndarray::{Array1, Array2};

struct MyProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl Problem for MyProblem {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        // Calculate residuals: model(x, params) - y_data
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| {
                // Model: y = a * exp(-b * x) + c
                let a = params[0];
                let b = params[1];
                let c = params[2];
                let model_value = a * (-b * x).exp() + c;
                
                // Residual: model_value - observed_value
                model_value - y
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        // Optional: Provide analytical Jacobian for better performance
        // If not implemented, finite differences will be used
        unimplemented!("Using automatic finite differences");
    }
    
    fn parameter_count(&self) -> usize {
        3  // Number of parameters in the model
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()  // Number of data points
    }
    
    fn has_jacobian(&self) -> bool {
        false  // Using automatic finite differences
    }
}
```

## Parameter System Integration

For more flexibility, you can use the parameter system with the LM algorithm:

```rust
use lmopt_rs::{LevenbergMarquardt, problem_params::{problem_from_parameter_problem, ParameterProblem}};
use lmopt_rs::parameters::Parameters;

// Define a problem using the parameter system
struct ParamProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    parameters: Parameters,
}

impl ParameterProblem for ParamProblem {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }
    
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }
    
    fn eval_with_parameters(&self) -> Result<Array1<f64>> {
        // Get parameter values
        let a = self.parameters.get("a").unwrap().value();
        let b = self.parameters.get("b").unwrap().value();
        let c = self.parameters.get("c").unwrap().value();
        
        // Calculate residuals
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| {
                a * (-b * x).exp() + c - y
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

// Convert to Problem trait and optimize
let adapter = problem_from_parameter_problem(&param_problem);
let result = optimizer.minimize(&adapter, param_problem.parameters_to_array()?)?;
```

## Convergence Control

The LM algorithm uses several convergence criteria:

1. **Function tolerance (ftol)**: Convergence is achieved when the relative reduction in the cost function is below this threshold.
2. **Parameter tolerance (xtol)**: Convergence is achieved when the relative change in parameters is below this threshold.
3. **Maximum iterations**: The maximum number of iterations to perform.

```rust
// Configure convergence criteria
let optimizer = LevenbergMarquardt::new()
    .with_ftol(1e-10)        // Stricter function tolerance
    .with_xtol(1e-8)         // Parameter tolerance
    .with_max_iterations(200); // Allow more iterations
```

## Handling Bounded Parameters

The Levenberg-Marquardt implementation in `lmopt-rs` handles parameter bounds through transformations:

```rust
// Define parameters with bounds
let mut params = Parameters::new();
params.add_param_with_bounds("a", 1.0, 0.0, 10.0)?;  // Between 0 and 10
params.add_param_with_bounds("b", 0.5, 0.0, f64::INFINITY)?;  // Positive only
params.add_param("c", 0.0)?;  // Unbounded

// Parameters will be transformed internally to ensure bounds are respected
```

## Advanced Algorithm Options

### Jacobian Calculation

The Jacobian can be calculated in several ways:

```rust
// 1. Using automatic finite differences (default)
let optimizer = LevenbergMarquardt::new();

// 2. Using a custom Jacobian method by implementing Problem::jacobian

// 3. Using autodiff (when the feature is enabled)
let optimizer = LevenbergMarquardt::new().with_differentiation_method(DiffMethod::AutoDiff);
```

### Linear Algebra Decomposition

Different decomposition methods can be used for solving the LM update equations:

```rust
use lmopt_rs::lm::DecompositionMethod;

// Select a specific decomposition method
let optimizer = LevenbergMarquardt::new()
    .with_decomposition_method(DecompositionMethod::Cholesky); // Faster but less stable
    
// Or use SVD for more robust behavior with ill-conditioned problems
let optimizer = LevenbergMarquardt::new()
    .with_decomposition_method(DecompositionMethod::SVD);
```

### Trust Region Adjustments

Control how the trust region (damping parameter) is adjusted:

```rust
let optimizer = LevenbergMarquardt::new()
    .with_lambda(0.001)           // Initial damping
    .with_lambda_up_factor(5.0)   // Increase factor when step fails
    .with_lambda_down_factor(0.2); // Decrease factor when step succeeds
```

## Performance Considerations

1. **Analytical Jacobian**: Whenever possible, implement the `jacobian` method for significant performance gains.
2. **Appropriate Initial Values**: Good initial parameter values can dramatically improve convergence.
3. **Scaling**: Ensure parameters are reasonably scaled (similar orders of magnitude).
4. **Residual Function**: Optimize the residual function calculation as it's the most frequently called operation.

## Compatibility

The `lmopt-rs` implementation is compatible with the `levenberg-marquardt` crate, allowing for easy migration:

```rust
// Use the compatibility feature
use lmopt_rs::lm_compat::LmProblemAdapter;

// Convert a problem from levenberg-marquardt to lmopt-rs
let lm_problem = /* your problem */;
let lmopt_problem = LmProblemAdapter::new(lm_problem);

// Solve using lmopt-rs
let result = optimizer.minimize(&lmopt_problem, initial_params)?;
```

## Common Issues and Solutions

### Poor Convergence

- Try different initial values
- Adjust the damping parameter (lambda)
- Check if the problem is well-conditioned
- Consider scaling your parameters

### Hitting Parameter Bounds

- Check if bounds are appropriate
- Consider reparameterizing your model
- Use looser bounds if physically reasonable

### Slow Performance

- Implement analytical Jacobian
- Optimize residual calculation
- Use a more efficient linear algebra decomposition method