# Basic Fitting Examples

This guide demonstrates how to perform basic curve fitting with `lmopt-rs`. We'll cover several common scenarios, from simple function fitting to more complex models with bounds and constraints.

## Related Documentation
- [Getting Started Guide](../getting_started.md)
- [Parameter System](../concepts/parameters.md)
- [Model System](../concepts/models.md)
- [Levenberg-Marquardt Algorithm](../concepts/lm_algorithm.md)
- [Uncertainty Analysis](../concepts/uncertainty.md)
- [Composite Models](./composite_models.md) - For more advanced model examples
- [Global Optimization](./global_optimization.md) - For complex fitting scenarios

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

## Advanced Parameter Expressions

You can use parameter expressions to define derived parameters that are calculated from other parameters:

```rust
use lmopt_rs::model::{fit, BaseModel, Model};
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for a Gaussian peak
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.iter().map(|&x| {
        // Gaussian with amplitude=3.0, center=0.0, sigma=1.0
        let true_value = 3.0 * (-0.5 * x.powi(2)).exp();
        let noise = rand::random::<f64>() * 0.1 - 0.05;
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create parameters with expressions
    let mut params = Parameters::new();
    
    // Basic parameters
    params.add_param_with_bounds("amplitude", 2.0, 0.0, 10.0)?;
    params.add_param("center", 0.0)?;
    params.add_param_with_bounds("sigma", 0.8, 0.1, 5.0)?;
    
    // Derived parameters using expressions
    // FWHM = 2.355 * sigma (Full width at half maximum)
    params.add_param_with_expr("fwhm", 0.0, "2.355 * sigma")?;
    
    // Area = amplitude * sigma * sqrt(2*pi)
    params.add_param_with_expr("area", 0.0, "amplitude * sigma * sqrt(2*pi)")?;
    
    // Height at peak = amplitude
    params.add_param_with_expr("height", 0.0, "amplitude")?;
    
    // Create the model
    let model_fn = |params: &Parameters, x: &Array1<f64>| {
        let amplitude = params.get("amplitude").unwrap().value();
        let center = params.get("center").unwrap().value();
        let sigma = params.get("sigma").unwrap().value();
        
        let result = x.iter()
            .map(|&x_val| {
                let arg = (x_val - center) / sigma;
                amplitude * (-0.5 * arg * arg).exp()
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(result))
    };
    
    let mut model = BaseModel::new(params, model_fn);
    
    // Fit the model
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Print results including derived parameters
    println!("Fit successful: {}", result.success);
    println!("Basic Parameters:");
    println!("  Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("sigma").unwrap().value());
    
    println!("\nDerived Parameters:");
    println!("  FWHM: {:.4}", model.parameters().get("fwhm").unwrap().value());
    println!("  Area: {:.4}", model.parameters().get("area").unwrap().value());
    println!("  Height: {:.4}", model.parameters().get("height").unwrap().value());
    
    println!("\nTrue parameters: amplitude=3.0, center=0.0, sigma=1.0");
    println!("True derived: fwhm=2.355, area=7.525, height=3.0");
    
    Ok(())
}
```

## Monte Carlo Uncertainty Analysis

For more robust uncertainty estimation, you can use Monte Carlo methods:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::GaussianModel;
use lmopt_rs::uncertainty::uncertainty_analysis_with_monte_carlo;
use ndarray::Array1;
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha8Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for a Gaussian peak
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.iter().map(|&x| {
        // Gaussian with amplitude=3.0, center=0.0, sigma=1.0
        let true_value = 3.0 * (-0.5 * x.powi(2)).exp();
        let noise = rand::thread_rng().gen_range(-0.2..0.2);
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create and fit the model
    let mut model = GaussianModel::new("", false);
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Create a seeded RNG for reproducibility
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    
    // Perform Monte Carlo uncertainty analysis
    let mc_result = uncertainty_analysis_with_monte_carlo(
        &model,
        &result,
        1000,       // Number of Monte Carlo samples
        &mut rng
    )?;
    
    // Print results
    println!("Fit Results:");
    println!("  Amplitude: {:.4}", model.parameters().get("amplitude").unwrap().value());
    println!("  Center: {:.4}", model.parameters().get("center").unwrap().value());
    println!("  Sigma: {:.4}", model.parameters().get("sigma").unwrap().value());
    
    println!("\nMonte Carlo Statistics:");
    for (name, dist) in &mc_result.parameter_distributions {
        println!("Parameter: {}", name);
        println!("  Mean: {:.4}", dist.mean());
        println!("  Std Dev: {:.4}", dist.std_dev());
        println!("  95% CI: [{:.4}, {:.4}]", 
                dist.percentile(2.5), dist.percentile(97.5));
    }
    
    // Check for parameter correlations
    println!("\nParameter Correlations:");
    println!("  Amplitude-Center: {:.4}", 
             mc_result.parameter_correlation("amplitude", "center").unwrap_or(0.0));
    println!("  Amplitude-Sigma: {:.4}", 
             mc_result.parameter_correlation("amplitude", "sigma").unwrap_or(0.0));
    println!("  Center-Sigma: {:.4}", 
             mc_result.parameter_correlation("center", "sigma").unwrap_or(0.0));
    
    Ok(())
}
```

## Curve Fitting with Predefined Step Functions

Here's an example of fitting a step-like function to data:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::StepModel;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data for a step function
    let x = Array1::linspace(-5.0, 5.0, 100);
    let y = x.iter().map(|&x| {
        // Step function with amplitude=2.0, center=0.0, sigma=0.5 (transition width)
        let true_value = 1.0 + 2.0 / (1.0 + (-(x - 0.0) / 0.5).exp());
        let noise = rand::random::<f64>() * 0.1 - 0.05;
        true_value + noise
    }).collect::<Vec<f64>>();
    let y = Array1::from_vec(y);
    
    // Create a step model
    let mut model = StepModel::new("", true);  // Empty prefix, with baseline
    
    // Set initial guesses
    model.parameters_mut().get_mut("amplitude").unwrap().set_value(1.5)?;
    model.parameters_mut().get_mut("center").unwrap().set_value(0.5)?;
    model.parameters_mut().get_mut("sigma").unwrap().set_value(1.0)?;
    model.parameters_mut().get_mut("baseline").unwrap().set_value(0.8)?;
    
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
    
    println!("\nTrue parameters: amplitude=2.0, center=0.0, sigma=0.5, baseline=1.0");
    
    // Analyze residuals
    let residuals = model.residuals(&x, &y)?;
    let rms_error = (residuals.iter().map(|r| r.powi(2)).sum::<f64>() / residuals.len() as f64).sqrt();
    println!("RMS Error: {:.6}", rms_error);
    
    Ok(())
}
```

## Fitting with Weighted Data

Sometimes, data points have different uncertainties. Here's how to implement weighted fitting:

```rust
use lmopt_rs::{LevenbergMarquardt, Problem};
use ndarray::{array, Array1, Array2, ArrayView1};

// Define a weighted problem
struct WeightedProblem {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    weights: Array1<f64>,  // Weights for each data point
}

impl Problem for WeightedProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Linear model: y = m*x + c
        let m = params[0];
        let c = params[1];
        
        // Calculate weighted residuals: (model - data) * sqrt(weight)
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .zip(self.weights.iter())
            .map(|((&x, &y), &w)| {
                let model = m * x + c;
                (model - y) * w.sqrt()  // Apply weight to residual
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn parameter_count(&self) -> usize {
        2  // m, c (slope and intercept)
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data with varying uncertainties
    let x = Array1::linspace(0.0, 10.0, 50);
    let mut rng = rand::thread_rng();
    
    // Generate y values with different noise levels
    let mut y = Array1::zeros(50);
    let mut weights = Array1::zeros(50);
    
    for i in 0..50 {
        let x_val = x[i];
        
        // True linear model: y = 2*x + 1
        let true_y = 2.0 * x_val + 1.0;
        
        // Different uncertainty for different regions
        let uncertainty = if x_val < 3.0 {
            0.5  // High uncertainty
        } else if x_val < 7.0 {
            0.2  // Medium uncertainty
        } else {
            0.1  // Low uncertainty
        };
        
        // Add noise proportional to uncertainty
        let noise = rng.gen_range(-uncertainty..uncertainty);
        y[i] = true_y + noise;
        
        // Weight is inversely proportional to variance
        weights[i] = 1.0 / (uncertainty * uncertainty);
    }
    
    // Create our weighted problem
    let problem = WeightedProblem {
        x_data: x.clone(),
        y_data: y.clone(),
        weights: weights.clone(),
    };
    
    // Create the optimizer
    let mut optimizer = LevenbergMarquardt::with_default_config();
    
    // Set initial parameter guess [m, c]
    let initial_params = array![1.0, 0.0];
    
    // Run the optimization
    let result = optimizer.minimize(&problem, initial_params)?;
    
    // Print results
    println!("Weighted fit successful: {}", result.success);
    println!("Parameters:");
    println!("  Slope (m): {:.4}", result.params[0]);
    println!("  Intercept (c): {:.4}", result.params[1]);
    println!("Cost: {:.6}", result.cost);
    
    // Compare with unweighted fit
    let unweighted_problem = WeightedProblem {
        x_data: x.clone(),
        y_data: y.clone(),
        weights: Array1::ones(50),  // Equal weights
    };
    
    let unweighted_result = optimizer.minimize(&unweighted_problem, initial_params)?;
    
    println!("\nUnweighted fit parameters:");
    println!("  Slope (m): {:.4}", unweighted_result.params[0]);
    println!("  Intercept (c): {:.4}", unweighted_result.params[1]);
    println!("Cost: {:.6}", unweighted_result.cost);
    
    println!("\nTrue parameters: m=2.0, c=1.0");
    
    Ok(())
}
```

These examples demonstrate the versatility and power of `lmopt-rs` for various fitting applications. For more advanced usage, see the [Composite Models](./composite_models.md) and [Global Optimization](./global_optimization.md) examples.