# lmopt-rs

A Rust implementation of the Levenberg-Marquardt algorithm for nonlinear least-squares optimization, with comprehensive uncertainty calculation capabilities.

[![Crate](https://img.shields.io/crates/v/lmopt-rs.svg)](https://crates.io/crates/lmopt-rs)
[![Documentation](https://docs.rs/lmopt-rs/badge.svg)](https://docs.rs/lmopt-rs)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ameyanagi.github.io/lmopt-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Levenberg-Marquardt Algorithm**: A powerful implementation compatible with the `levenberg-marquardt` crate
- **Comprehensive Uncertainty Analysis**: Similar to lmfit-py, including:
  - Covariance matrix estimation
  - Standard error calculations
  - Confidence intervals
  - Monte Carlo uncertainty propagation
- **Parameter System**: Flexible parameter handling with:
  - Bounds constraints (min/max)
  - Algebraic constraints between parameters
  - Parameter expressions
  - Fixed/variable parameters
- **Model System**: Hierarchical model architecture with:
  - Built-in models for common fitting problems
  - Composite model support
  - Custom model creation
- **Global Optimization**: Multiple methods for finding global minima:
  - Simulated Annealing
  - Differential Evolution
  - Basin Hopping
  - Hybrid methods

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
lmopt-rs = "0.1.0"
```

## Basic Usage

### Simple Problem Definition

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::{array, Array1};

// Define a problem that implements the Problem trait
struct QuadraticProblem;

impl Problem for QuadraticProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // params[0] = a, params[1] = b, params[2] = c
        // We're fitting: y = a*x^2 + b*x + c
        
        // Sample data points for (x, y): (1, 6), (2, 11), (3, 18)
        let x_values = array![1.0, 2.0, 3.0];
        let y_values = array![6.0, 11.0, 18.0];
        
        // Calculate residuals: y_model - y_data
        let residuals = x_values.mapv(|x| {
            params[0] * x.powi(2) + params[1] * x + params[2]
        }) - y_values;
        
        Ok(residuals)
    }
    
    fn parameter_count(&self) -> usize {
        3  // a, b, c
    }
    
    fn residual_count(&self) -> usize {
        3  // Number of data points
    }
}

fn main() -> lmopt_rs::Result<()> {
    // Create the problem
    let problem = QuadraticProblem;
    
    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Initial guess for parameters [a, b, c]
    let initial_params = array![1.0, 1.0, 1.0];
    
    // Run the optimization
    let result = optimizer.minimize(&problem, initial_params)?;
    
    println!("Optimization successful: {}", result.success);
    println!("Final parameters: {:?}", result.params);
    println!("Cost (sum of squared residuals): {}", result.cost);
    println!("Number of iterations: {}", result.iterations);
    
    Ok(())
}
```

### Using the Parameter System

```rust
use lmopt_rs::{LevenbergMarquardt, ParameterProblem, parameters::Parameters};
use lmopt_rs::problem_params::problem_from_parameter_problem;
use ndarray::{Array1, array};

struct GaussianProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    parameters: Parameters,
}

impl GaussianProblem {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        let mut parameters = Parameters::new();
        parameters.add_param_with_bounds("amplitude", 1.0, 0.0, f64::INFINITY).unwrap();
        parameters.add_param("center", 0.0).unwrap();
        parameters.add_param_with_bounds("sigma", 1.0, 0.01, f64::INFINITY).unwrap();
        
        // Add a derived parameter for FWHM (Full Width at Half Maximum)
        parameters.add_param_with_expr("fwhm", 2.355, "2.355 * sigma").unwrap();
        
        Self {
            x_data,
            y_data,
            parameters,
        }
    }
    
    fn gaussian(x: f64, amplitude: f64, center: f64, sigma: f64) -> f64 {
        amplitude * (-((x - center).powi(2)) / (2.0 * sigma.powi(2))).exp()
    }
}

impl ParameterProblem for GaussianProblem {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }
    
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }
    
    fn eval_with_parameters(&self) -> lmopt_rs::Result<Array1<f64>> {
        let amplitude = self.parameters.get("amplitude").unwrap().value();
        let center = self.parameters.get("center").unwrap().value();
        let sigma = self.parameters.get("sigma").unwrap().value();
        
        // Calculate residuals
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| Self::gaussian(x, amplitude, center, sigma) - y)
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

fn main() -> lmopt_rs::Result<()> {
    // Create some test data
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.mapv(|x_val| {
        // true parameters: amplitude=3.0, center=1.0, sigma=0.5
        3.0 * (-((x_val - 1.0).powi(2)) / (2.0 * 0.5.powi(2))).exp() 
    });
    
    // Create our problem with initial guesses
    let mut problem = GaussianProblem::new(x, y);
    
    // Create the adapter for the optimizer
    let adapter = problem_from_parameter_problem(&problem);
    
    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Get initial parameters as array
    let initial_params = problem.parameters_to_array()?;
    
    // Run the optimization
    let result = optimizer.minimize(&adapter, initial_params)?;
    
    // Update the model parameters with optimized values
    problem.update_parameters_from_array(&result.params)?;
    
    // Print results
    println!("Optimization successful: {}", result.success);
    println!("Amplitude: {:.4}", problem.parameters().get("amplitude").unwrap().value());
    println!("Center: {:.4}", problem.parameters().get("center").unwrap().value());
    println!("Sigma: {:.4}", problem.parameters().get("sigma").unwrap().value());
    println!("FWHM: {:.4}", problem.parameters().get("fwhm").unwrap().value());
    
    Ok(())
}
```

### Using the Model System

```rust
use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::models::ExponentialModel;
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate some data for an exponential decay
    let x = Array1::linspace(0.0, 10.0, 50);
    let y = x.mapv(|x_val| 5.0 * (-x_val / 2.5).exp() + 1.0);
    
    // Create and fit an exponential model
    let mut exp_model = ExponentialModel::new("", true);
    let result = fit(&mut exp_model, x.clone(), y.clone())?;
    
    // Print results
    println!("Fit success: {}", result.success);
    println!("Parameters:");
    println!("  amplitude = {:.3}", exp_model.parameters().get("amplitude").unwrap().value());
    println!("  decay = {:.3}", exp_model.parameters().get("decay").unwrap().value());
    println!("  baseline = {:.3}", exp_model.parameters().get("baseline").unwrap().value());
    println!("Cost (residual sum): {:.6}", result.cost);
    
    // Create a custom model with user-defined function
    let mut params = Parameters::new();
    params.add_param("a", 1.0)?;
    params.add_param("b", 1.0)?;
    params.add_param("c", 0.0)?;
    
    let custom_model = BaseModel::new(params, |params, x| {
        // Custom function: y = a*sin(b*x + c)
        let a = params.get("a").unwrap().value();
        let b = params.get("b").unwrap().value();
        let c = params.get("c").unwrap().value();
        
        let result = x.iter()
            .map(|&x_val| a * (b * x_val + c).sin())
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(result))
    });
    
    Ok(())
}
```

### Uncertainty Analysis

```rust
use lmopt_rs::{
    uncertainty_analysis, uncertainty_analysis_with_monte_carlo,
    covariance_matrix, standard_errors, monte_carlo_refit,
    ConfidenceInterval, UncertaintyResult
};
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::GaussianModel;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate data for a Gaussian peak
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.mapv(|x_val| {
        // true parameters: amplitude=3.0, center=0.0, sigma=1.0
        3.0 * (-x_val.powi(2) / 2.0).exp()
    });
    
    // Create and fit a Gaussian model
    let mut model = GaussianModel::new("", true);
    let fit_result = fit(&mut model, x.clone(), y.clone())?;
    
    // Basic uncertainty analysis
    let uncertainty = uncertainty_analysis(&model, &fit_result)?;
    
    // Print confidence intervals
    println!("Parameter confidence intervals (95%):");
    for (name, interval) in &uncertainty.confidence_intervals {
        println!("  {}: {:.4} ± {:.4}", name, interval.value, interval.error);
        println!("     [{:.4}, {:.4}]", interval.lower, interval.upper);
    }
    
    // Perform Monte Carlo analysis for more robust uncertainty estimation
    let monte_carlo = uncertainty_analysis_with_monte_carlo(&model, &fit_result, 1000)?;
    
    // Print Monte Carlo confidence intervals
    println!("\nMonte Carlo confidence intervals (95%):");
    for (name, dist) in &monte_carlo.parameter_distributions {
        let p5 = dist.percentile(5.0);
        let p95 = dist.percentile(95.0);
        let mean = dist.mean();
        
        println!("  {}: {:.4} [{:.4}, {:.4}]", name, mean, p5, p95);
    }
    
    Ok(())
}
```

### Global Optimization

```rust
use lmopt_rs::{
    optimize_global, optimize_global_param_problem, 
    parameters::Parameters, BasinHopping, DifferentialEvolution, 
    GlobalOptResult, GlobalOptimizer, HybridGlobal, ParameterProblem, 
    Problem, SimulatedAnnealing
};
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create the problem
    let problem = MultiWellProblem;
    
    // Define bounds for the parameters
    let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
    
    // Try different global optimization methods
    
    // 1. Simulated Annealing
    let sa_optimizer = SimulatedAnnealing::new();
    let sa_result = sa_optimizer.optimize(&problem, &bounds, 1000, 100, 1e-6)?;
    
    println!("Simulated Annealing result:");
    println!("  Success: {}", sa_result.success);
    println!("  Solution: ({:.6}, {:.6})", sa_result.params[0], sa_result.params[1]);
    println!("  Cost: {:.6e}", sa_result.cost);
    
    // 2. Differential Evolution
    let de_optimizer = DifferentialEvolution::new();
    let de_result = de_optimizer.optimize(&problem, &bounds, 100, 20, 1e-6)?;
    
    println!("\nDifferential Evolution result:");
    println!("  Success: {}", de_result.success);
    println!("  Solution: ({:.6}, {:.6})", de_result.params[0], de_result.params[1]);
    println!("  Cost: {:.6e}", de_result.cost);
    
    // 3. Hybrid approach (global + local refinement)
    let hybrid_optimizer = HybridGlobal::new();
    let hybrid_result = hybrid_optimizer.optimize(&problem, &bounds, 50, 10, 1e-6)?;
    
    println!("\nHybrid Global result:");
    println!("  Success: {}", hybrid_result.success);
    println!("  Solution: ({:.6}, {:.6})", hybrid_result.params[0], hybrid_result.params[1]);
    println!("  Cost: {:.6e}", hybrid_result.cost);
    
    Ok(())
}
```

## Documentation

Comprehensive documentation is available at [docs.rs/lmopt-rs](https://docs.rs/lmopt-rs).

## Features

- `matrix`: Enable matrix operations using faer and faer-ext (enabled by default)
- `lm`: Enable Levenberg-Marquardt implementation (enabled by default)
- `lm-compat`: Enable compatibility with the levenberg-marquardt crate
- `autodiff`: Enable experimental autodiff support for derivative calculation
- `parameters`: Enable just the parameter system

## References and Compatibility

- Compatible with the [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt) crate
- Inspired by [lmfit-py](https://lmfit.github.io/lmfit-py/) for uncertainty calculations
- Uses [faer](https://docs.rs/faer/latest/faer/) for high-performance matrix operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.