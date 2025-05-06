# Getting Started with lmopt-rs

This guide will help you get started with `lmopt-rs`, a powerful library for nonlinear least-squares optimization with the Levenberg-Marquardt algorithm and uncertainty analysis.

## Installation

Add `lmopt-rs` to your `Cargo.toml`:

```toml
[dependencies]
lmopt-rs = "0.1.0"
```

For specific features, you can use:

```toml
[dependencies]
lmopt-rs = { version = "0.1.0", features = ["lm-compat"] }
```

Available features:
- `matrix`: Matrix operations using faer and faer-ext (enabled by default)
- `lm`: Levenberg-Marquardt implementation (enabled by default)
- `lm-compat`: Compatibility with the levenberg-marquardt crate
- `autodiff`: Experimental autodiff support for derivative calculation
- `parameters`: Just the parameter system

## Basic Concepts

The `lmopt-rs` library is built around a few core concepts:

1. **Problem**: Defines a least-squares optimization problem
2. **Parameters**: Manages parameters with constraints and expressions
3. **Model**: Provides pre-built models for common fitting tasks
4. **Uncertainty Analysis**: Calculates confidence intervals and error estimates
5. **Global Optimization**: Methods for finding global minima

## Quick Examples

### Basic Problem Definition

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::{array, Array1};

// Define your problem by implementing the Problem trait
struct MyProblem;

impl Problem for MyProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Calculate residuals: model(x, params) - y
        // For a simple y = a*x^2 + b*x + c model
        
        // Sample data points: (x, y)
        let x_values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_values = array![2.0, 5.0, 10.0, 17.0, 26.0];
        
        // Calculate residuals
        let residuals = x_values.mapv(|x| {
            params[0] * x.powi(2) + params[1] * x + params[2] - y_actual(x)
        });
        
        Ok(residuals)
    }
    
    fn parameter_count(&self) -> usize {
        3  // Number of parameters (a, b, c)
    }
    
    fn residual_count(&self) -> usize {
        5  // Number of data points
    }
}

fn main() -> lmopt_rs::Result<()> {
    let problem = MyProblem;
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Initial parameter guess [a, b, c]
    let initial_params = array![0.0, 0.0, 0.0];
    
    // Run the optimization
    let result = optimizer.minimize(&problem, initial_params)?;
    
    println!("Optimized parameters: {:?}", result.params);
    println!("Cost: {}", result.cost);
    
    Ok(())
}
```

### Using Built-in Models

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::GaussianModel;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate x data and y data
    let x_data = Array1::linspace(-5.0, 5.0, 100);
    let y_data = x_data.mapv(|x| 3.0 * (-x.powi(2) / 2.0).exp() + 0.5);
    
    // Create a Gaussian model
    let mut model = GaussianModel::new("", true);
    
    // Fit the model to the data
    let result = fit(&mut model, x_data.clone(), y_data.clone())?;
    
    // Print the results
    println!("Fit success: {}", result.success);
    println!("Parameters:");
    println!("  amplitude = {:.3}", model.parameters().get("amplitude").unwrap().value());
    println!("  center = {:.3}", model.parameters().get("center").unwrap().value());
    println!("  sigma = {:.3}", model.parameters().get("sigma").unwrap().value());
    println!("  baseline = {:.3}", model.parameters().get("baseline").unwrap().value());
    
    Ok(())
}
```

## Next Steps

- Learn more about the [Parameter System](./concepts/parameters.md)
- Explore the [Model System](./concepts/models.md)
- Understand [Uncertainty Analysis](./concepts/uncertainty.md)
- Try [Global Optimization](./concepts/global_optimization.md)
- Check examples in the [examples directory](../examples/)

For more detailed information, refer to the [API documentation](https://docs.rs/lmopt-rs).