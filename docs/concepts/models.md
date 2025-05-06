# Model System

The model system in `lmopt-rs` provides a flexible framework for defining and fitting mathematical models to data. It's inspired by the model handling in lmfit-py and offers an intuitive way to create, combine, and fit various types of models.

## Core Concepts

The model system is built around these key components:

1. **Model Trait**: Defines the core interface for all models
2. **BaseModel**: A generic implementation for user-defined models
3. **Built-in Models**: Common predefined models like Gaussian, exponential, etc.
4. **Composite Models**: Combine multiple models with operations like addition or multiplication

## Using Built-in Models

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, ExponentialModel, LinearModel};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data
    let x = Array1::linspace(0.0, 10.0, 100);
    let y = x.mapv(|x_val| {
        // Exponential decay
        5.0 * (-x_val / 3.0).exp() + 0.5
    });
    
    // Create an exponential model with baseline
    let mut model = ExponentialModel::new("", true);
    
    // Fit the model to the data
    let result = fit(&mut model, x.clone(), y.clone())?;
    
    // Print results
    println!("Fit success: {}", result.success);
    println!("Parameters:");
    println!("  Amplitude: {:.3}", model.parameters().get("amplitude").unwrap().value());
    println!("  Decay: {:.3}", model.parameters().get("decay").unwrap().value());
    println!("  Baseline: {:.3}", model.parameters().get("baseline").unwrap().value());
    
    Ok(())
}
```

## Available Built-in Models

The library provides several commonly used models:

### Peak Models
- **GaussianModel**: `f(x) = amplitude * exp(-(x-center)²/(2*sigma²)) + baseline`
- **LorentzianModel**: `f(x) = amplitude * gamma²/((x-center)²+gamma²) + baseline`
- **VoigtModel**: Convolution of Gaussian and Lorentzian profiles
- **PseudoVoigtModel**: Linear combination of Gaussian and Lorentzian

### Background Models
- **ConstantModel**: `f(x) = c`
- **LinearModel**: `f(x) = c0 + c1*x`
- **QuadraticModel**: `f(x) = c0 + c1*x + c2*x²`
- **PolynomialModel**: `f(x) = c0 + c1*x + c2*x² + ... + cn*x^n`

### Special Models
- **ExponentialModel**: `f(x) = amplitude * exp(-x/decay) + baseline`
- **PowerLawModel**: `f(x) = amplitude * x^exponent + baseline`
- **SigmoidModel**: `f(x) = amplitude / (1 + exp(-(x-center)/sigma)) + baseline`
- **StepModel**: Step functions (linear, arctan, erf, etc.)

## Creating Custom Models

You can easily create custom models using the `BaseModel` implementation:

```rust
use lmopt_rs::model::{BaseModel, Model};
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define parameters for a sine wave model
    let mut params = Parameters::new();
    params.add_param("amplitude", 1.0)?;
    params.add_param("frequency", 1.0)?;
    params.add_param("phase", 0.0)?;
    params.add_param("offset", 0.0)?;
    
    // Create a custom model with a closure
    let sine_model = BaseModel::new(params, |params, x| {
        let amplitude = params.get("amplitude").unwrap().value();
        let frequency = params.get("frequency").unwrap().value();
        let phase = params.get("phase").unwrap().value();
        let offset = params.get("offset").unwrap().value();
        
        // Calculate the model values
        let result = x.iter()
            .map(|&x_val| {
                offset + amplitude * (frequency * x_val + phase).sin()
            })
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(result))
    });
    
    // Use the model for fitting...
    
    Ok(())
}
```

## Composite Models

One of the most powerful features is the ability to combine models to create more complex functions:

```rust
use lmopt_rs::model::{Model};
use lmopt_rs::models::{GaussianModel, LinearModel, add, multiply};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create individual models
    let peak1 = GaussianModel::new("g1_", true);
    let peak2 = GaussianModel::new("g2_", true);
    let background = LinearModel::new("bg_", false);
    
    // Create a composite model: two peaks plus a linear background
    let two_peaks = add(peak1, peak2, None, None)?;
    let full_model = add(two_peaks, background, None, None)?;
    
    // Use for fitting...
    
    Ok(())
}
```

### Available Composite Operations

- `add(model1, model2, ...)`: Addition of multiple models
- `multiply(model1, model2, ...)`: Multiplication of models
- `subtract(model1, model2)`: Subtraction of models
- `divide(model1, model2)`: Division of models
- `composite_with_shared_params(model1, model2, ...)`: Create composite models with shared parameters

## Shared Parameters

You can create models that share certain parameters:

```rust
use lmopt_rs::models::composite_with_shared_params;
use std::collections::HashMap;

// Create a map of shared parameters
let mut shared_params = HashMap::new();
shared_params.insert("g1_sigma".to_string(), "g2_sigma".to_string());

// Create a composite model with shared parameters
let shared_model = composite_with_shared_params(
    gaussian1, 
    gaussian2, 
    Operation::Add, 
    shared_params
)?;
```

This is particularly useful for multi-peak fitting where you want to enforce that all peaks have the same width.

## Model Fitting

The `fit` function provides a convenient way to fit models to data:

```rust
use lmopt_rs::model::{fit, Model};

let result = fit(&mut model, x_data, y_data)?;

// Access fit results
println!("Success: {}", result.success);
println!("Cost (sum of squared residuals): {}", result.cost);
println!("Iterations: {}", result.iterations);

// Access standard errors
for (name, error) in &result.standard_errors {
    println!("{}: {:.4} ± {:.4}", name, 
             model.parameters().get(name).unwrap().value(), 
             error);
}
```

## Custom Fitting

For more control over the fitting process, you can use the lower-level APIs:

```rust
use lmopt_rs::{LevenbergMarquardt, problem_params::problem_from_parameter_problem};

// Create adapter for the optimizer
let adapter = problem_from_parameter_problem(&model);

// Configure the optimizer
let mut optimizer = LevenbergMarquardt::new()
    .with_max_iterations(100)
    .with_ftol(1e-10)
    .with_xtol(1e-10);

// Get initial parameters as array
let initial_params = model.parameters_to_array()?;

// Run optimization
let result = optimizer.minimize(&adapter, initial_params)?;

// Update model parameters with optimized values
model.update_parameters_from_array(&result.params)?;
```

## Model Initialization

Models can provide smart parameter initialization based on data:

```rust
// Initialize peak parameters from data
model.guess_parameters(&x, &y)?;

// Or manually set initial values
model.parameters_mut().get_mut("amplitude").unwrap().set_value(3.0)?;
model.parameters_mut().get_mut("center").unwrap().set_value(0.0)?;
```

## Advanced Usage

### Model Evaluation

```rust
// Evaluate model at specific x values
let y_model = model.eval(&x)?;

// Calculate residuals
let residuals = model.eval_with_parameters()?;
```

### Model Serialization

```rust
// Save model to JSON
let json = serde_json::to_string(&model)?;

// Load model from JSON
let loaded_model: MyModel = serde_json::from_str(&json)?;
```

### Prefixes and Parameter Naming

```rust
// Create a model with a specific prefix
let model = GaussianModel::new("peak1_", true);

// Access parameters with prefix
let value = model.parameters().get("peak1_amplitude").unwrap().value();
```

### Custom Model Implementations

For even more control, you can implement the `Model` trait directly:

```rust
use lmopt_rs::model::Model;
use lmopt_rs::parameters::Parameters;
use ndarray::Array1;

struct CustomModel {
    parameters: Parameters,
    // other fields...
}

impl Model for CustomModel {
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }
    
    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }
    
    fn eval(&self, x: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Custom evaluation logic
        // ...
        
        Ok(result)
    }
    
    // Optionally override other methods...
}
```