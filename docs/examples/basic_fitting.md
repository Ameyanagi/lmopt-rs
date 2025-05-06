# Basic Fitting Examples

This guide demonstrates how to perform basic curve fitting with `lmopt-rs`. We'll cover several common scenarios, from simple function fitting to more complex models with bounds and constraints.

## Simple Function Fitting

Let's start with a basic example: fitting a quadratic function to data points.

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::{array, Array1, Array2};

// Define our problem: fitting y = a*x^2 + b*x + c
struct QuadraticProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl Problem for QuadraticProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // params[0] = a, params[1] = b, params[2] = c
        let a = params[0];
        let b = params[1];
        let c = params[2];
        
        // Calculate residuals: model(x) - y_data
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| {
                let y_model = a * x.powi(2) + b * x + c;
                y_model - y
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn parameter_count(&self) -> usize {
        3  // a, b, c
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some synthetic data with noise
    let x = Array1::linspace(-5.0, 5.0, 50);
    let y = x.iter().map(|&x| {
        // True function: y = 2*x^2 - 3*x + 1 + noise
        let true_value = 2.0 * x.powi(2) - 3.0 * x + 1.0;
        let noise = rand::random::<f64>() * 2.0 - 1.0;  // Random noise between -1 and 1
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create our problem
    let problem = QuadraticProblem {
        x_data: x.clone(),
        y_data: y.clone(),
    };
    
    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Set initial parameter guess [a, b, c]
    let initial_params = array![1.0, 1.0, 0.0];
    
    // Run the optimization
    let result = optimizer.minimize(&problem, initial_params)?;
    
    // Print results
    println!("Optimization successful: {}", result.success);
    println!("Final parameters: a={:.4}, b={:.4}, c={:.4}", 
             result.params[0], result.params[1], result.params[2]);
    println!("Cost (sum of squared residuals): {:.6}", result.cost);
    println!("Number of iterations: {}", result.iterations);
    
    // Compare with true values
    println!("\nTrue parameters: a=2.0, b=-3.0, c=1.0");
    
    Ok(())
}
```

## Using the Parameter System

Now let's use the parameter system for a more structured approach:

```rust
use lmopt_rs::{LevenbergMarquardt, ParameterProblem, problem_params::problem_from_parameter_problem};
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

// Define a model with parameters
struct ExponentialDecay {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    parameters: Parameters,
}

impl ExponentialDecay {
    fn new(x_data: Array1<f64>, y_data: Array1<f64>) -> Self {
        let mut parameters = Parameters::new();
        
        // Add parameters with meaningful names
        parameters.add_param_with_bounds("amplitude", 5.0, 0.0, f64::INFINITY).unwrap();
        parameters.add_param_with_bounds("decay_rate", 0.5, 0.0, f64::INFINITY).unwrap();
        parameters.add_param("offset", 1.0).unwrap();
        
        Self {
            x_data,
            y_data,
            parameters,
        }
    }
}

impl ParameterProblem for ExponentialDecay {
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }
    
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }
    
    fn eval_with_parameters(&self) -> lmopt_rs::Result<Array1<f64>> {
        // Get current parameter values
        let amplitude = self.parameters.get("amplitude").unwrap().value();
        let decay_rate = self.parameters.get("decay_rate").unwrap().value();
        let offset = self.parameters.get("offset").unwrap().value();
        
        // Calculate residuals: model(x) - y_data
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| {
                // Model: y = amplitude * exp(-decay_rate * x) + offset
                let y_model = amplitude * (-decay_rate * x).exp() + offset;
                y_model - y
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some synthetic data
    let x = Array1::linspace(0.0, 10.0, 50);
    let y = x.iter().map(|&x| {
        // True function: y = 10 * exp(-0.3 * x) + 2 + noise
        let true_value = 10.0 * (-0.3 * x).exp() + 2.0;
        let noise = rand::random::<f64>() * 0.4 - 0.2;  // Random noise between -0.2 and 0.2
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create our model
    let mut model = ExponentialDecay::new(x, y);
    
    // Create a problem adapter
    let adapter = problem_from_parameter_problem(&model);
    
    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Get initial parameters as array
    let initial_params = model.parameters_to_array()?;
    
    // Run the optimization
    let result = optimizer.minimize(&adapter, initial_params)?;
    
    // Update the model parameters with the optimized values
    model.update_parameters_from_array(&result.params)?;
    
    // Print results
    println!("Optimization successful: {}", result.success);
    println!("Parameters:");
    println!("  Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
    println!("  Decay rate: {:.4}", model.parameters().get("decay_rate").unwrap().value());
    println!("  Offset: {:.4}", model.parameters().get("offset").unwrap().value());
    println!("Cost: {:.6}", result.cost);
    println!("Iterations: {}", result.iterations);
    
    println!("\nTrue parameters: amplitude=10.0, decay_rate=0.3, offset=2.0");
    
    Ok(())
}
```

## Using Built-in Models

`lmopt-rs` provides built-in models for common functions:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::GaussianModel;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create some synthetic data for a Gaussian peak
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.iter().map(|&x| {
        // True function: Gaussian with amplitude=3.0, center=1.0, sigma=0.8, baseline=0.5
        let arg = (x - 1.0) / 0.8;
        let true_value = 3.0 * (-0.5 * arg * arg).exp() + 0.5;
        let noise = rand::random::<f64>() * 0.2 - 0.1;  // Random noise
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create a Gaussian model
    let mut model = GaussianModel::new("", true);  // Empty prefix, with baseline
    
    // Fit the model
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Print results
    println!("Fit successful: {}", result.success);
    println!("Parameters:");
    println!("  Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("sigma").unwrap().value());
    println!("  Baseline: {:.4}", model.parameters().get("baseline").unwrap().value());
    println!("Cost: {:.6}", result.cost);
    
    // Calculate standard errors
    println!("\nStandard Errors:");
    for (name, error) in &result.standard_errors {
        println!("  {}: {:.4}", name, error);
    }
    
    println!("\nTrue parameters: amplitude=3.0, center=1.0, sigma=0.8, baseline=0.5");
    
    Ok(())
}
```

## Fitting with Bounds and Constraints

Let's fit a model with parameter bounds and constraints:

```rust
use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for a sine wave with exponential decay
    let x = Array1::linspace(0.0, 20.0, 100);
    let y = x.iter().map(|&x| {
        // Model: amplitude * exp(-decay * x) * sin(frequency * x + phase) + offset
        let amplitude = 5.0;
        let decay = 0.1;
        let frequency = 1.0;
        let phase = 0.5;
        let offset = 1.0;
        
        let true_value = amplitude * (-decay * x).exp() * (frequency * x + phase).sin() + offset;
        let noise = rand::random::<f64>() * 0.2 - 0.1;
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create parameters with bounds and constraints
    let mut params = Parameters::new();
    
    // Parameters with bounds
    params.add_param_with_bounds("amplitude", 1.0, 0.0, 10.0)?;
    params.add_param_with_bounds("decay", 0.05, 0.0, 1.0)?;
    params.add_param_with_bounds("frequency", 0.5, 0.1, 5.0)?;
    params.add_param_with_bounds("phase", 0.0, -3.14, 3.14)?;
    params.add_param("offset", 0.0)?;
    
    // Add a constraint: frequency must be positive
    params.get_mut("frequency").unwrap().set_min(0.0)?;
    
    // Create a custom model
    let model_fn = |params: &Parameters, x: &Array1<f64>| {
        let amplitude = params.get("amplitude").unwrap().value();
        let decay = params.get("decay").unwrap().value();
        let frequency = params.get("frequency").unwrap().value();
        let phase = params.get("phase").unwrap().value();
        let offset = params.get("offset").unwrap().value();
        
        let result = x.iter()
            .map(|&x_val| {
                amplitude * (-decay * x_val).exp() * (frequency * x_val + phase).sin() + offset
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(result))
    };
    
    let mut model = BaseModel::new(params, model_fn);
    
    // Fit the model
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Print results
    println!("Fit successful: {}", result.success);
    println!("Parameters:");
    println!("  Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
    println!("  Decay: {:.4}", model.parameters().get("decay").unwrap().value());
    println!("  Frequency: {:.4}", model.parameters().get("frequency").unwrap().value());
    println!("  Phase: {:.4}", model.parameters().get("phase").unwrap().value());
    println!("  Offset: {:.4}", model.parameters().get("offset").unwrap().value());
    println!("Cost: {:.6}", result.cost);
    
    println!("\nTrue parameters: amplitude=5.0, decay=0.1, frequency=1.0, phase=0.5, offset=1.0");
    
    Ok(())
}
```

## Multiple Peak Fitting

Fitting multiple peaks to spectral data:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, LinearModel, add};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for a spectrum with multiple peaks
    let x = Array1::linspace(0.0, 10.0, 200);
    let y = x.iter().map(|&x| {
        // Background: linear with intercept=1.0, slope=0.1
        let background = 1.0 + 0.1 * x;
        
        // Peak 1: Gaussian with amplitude=5.0, center=2.0, sigma=0.5
        let peak1_arg = (x - 2.0) / 0.5;
        let peak1 = 5.0 * (-0.5 * peak1_arg * peak1_arg).exp();
        
        // Peak 2: Gaussian with amplitude=3.0, center=5.0, sigma=0.8
        let peak2_arg = (x - 5.0) / 0.8;
        let peak2 = 3.0 * (-0.5 * peak2_arg * peak2_arg).exp();
        
        // Peak 3: Gaussian with amplitude=4.0, center=8.0, sigma=0.6
        let peak3_arg = (x - 8.0) / 0.6;
        let peak3 = 4.0 * (-0.5 * peak3_arg * peak3_arg).exp();
        
        // Combine all components with noise
        let true_value = background + peak1 + peak2 + peak3;
        let noise = rand::random::<f64>() * 0.3 - 0.15;
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create individual models
    let mut peak1 = GaussianModel::new("p1_", false);  // No baseline
    peak1.parameters_mut().get_mut("p1_center").unwrap().set_value(2.0)?;
    
    let mut peak2 = GaussianModel::new("p2_", false);
    peak2.parameters_mut().get_mut("p2_center").unwrap().set_value(5.0)?;
    
    let mut peak3 = GaussianModel::new("p3_", false);
    peak3.parameters_mut().get_mut("p3_center").unwrap().set_value(8.0)?;
    
    let mut background = LinearModel::new("bg_", false);
    
    // Combine models
    let model1 = add(peak1, peak2, None, None)?;
    let model2 = add(model1, peak3, None, None)?;
    let mut model = add(model2, background, None, None)?;
    
    // Fit the model
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Print results
    println!("Fit successful: {}", result.success);
    println!("Parameters:");
    
    // Peak 1
    println!("Peak 1:");
    println!("  Amplitude: {:.4}", model.parameters().get("p1_amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("p1_center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("p1_sigma").unwrap().value());
    
    // Peak 2
    println!("Peak 2:");
    println!("  Amplitude: {:.4}", model.parameters().get("p2_amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("p2_center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("p2_sigma").unwrap().value());
    
    // Peak 3
    println!("Peak 3:");
    println!("  Amplitude: {:.4}", model.parameters().get("p3_amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("p3_center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("p3_sigma").unwrap().value());
    
    // Background
    println!("Background:");
    println!("  Intercept: {:.4}", model.parameters().get("bg_c0").unwrap().value());
    println!("  Slope: {:.4}", model.parameters().get("bg_c1").unwrap().value());
    
    println!("Cost: {:.6}", result.cost);
    
    Ok(())
}
```

## Uncertainty Analysis

Calculate confidence intervals for the fitted parameters:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::ExponentialModel;
use lmopt_rs::uncertainty::uncertainty_analysis;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for an exponential decay
    let x = Array1::linspace(0.0, 10.0, 50);
    let y = x.iter().map(|&x| {
        // Model: y = 5.0 * exp(-0.5 * x) + 1.0
        let true_value = 5.0 * (-0.5 * x).exp() + 1.0;
        let noise = rand::random::<f64>() * 0.3 - 0.15;
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create and fit the model
    let mut model = ExponentialModel::new("", true);  // Empty prefix, with baseline
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Calculate uncertainty
    let uncertainty = uncertainty_analysis(&model, &result)?;
    
    // Print results with confidence intervals
    println!("Fit Results with 95% Confidence Intervals:");
    
    for (name, interval) in &uncertainty.confidence_intervals {
        println!("{}: {:.4} ± {:.4}", name, 
                 model.parameters().get(name).unwrap().value(), 
                 interval.error);
        println!("   95% CI: [{:.4}, {:.4}]", interval.lower, interval.upper);
    }
    
    // Calculate goodness of fit metrics
    let residuals = model.eval_with_parameters()?;
    let ssr = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let sst = y.iter().map(|&y_i| (y_i - mean_y).powi(2)).sum::<f64>();
    let r_squared = 1.0 - ssr / sst;
    
    println!("\nGoodness of Fit:");
    println!("  R-squared: {:.4}", r_squared);
    println!("  Sum of squared residuals: {:.4}", ssr);
    
    Ok(())
}
```

These examples demonstrate the versatility and power of `lmopt-rs` for various fitting applications. For more advanced usage, check the other example documentation.