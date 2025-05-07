# Composite Models

This guide explains how to use composite models in `lmopt-rs` to fit complex data that requires multiple component models.

## Related Documentation
- [Getting Started Guide](../getting_started.md)
- [Parameter System](../concepts/parameters.md)
- [Model System](../concepts/models.md)
- [Uncertainty Analysis](../concepts/uncertainty.md)
- [Basic Fitting](./basic_fitting.md) - Start here for simpler fitting examples
- [Global Optimization](./global_optimization.md) - For finding global minima in complex models

## What Are Composite Models?

Composite models allow you to combine multiple simpler models to fit more complex data patterns. This is particularly useful for:

- Spectra with multiple peaks
- Signals with multiple components
- Data with baseline and multiple features
- Any situation where a single model function is insufficient

## Basic Composite Model Operations

`lmopt-rs` supports two basic operations for combining models:

1. **Addition**: Combines models by adding their outputs (e.g., multiple peaks)
2. **Multiplication**: Combines models by multiplying their outputs (e.g., applying a modulation)

### Addition Example

Adding two Gaussian peaks:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, add};
use ndarray::Array1;

// Create individual models
let mut g1 = GaussianModel::new("g1_", false);
let mut g2 = GaussianModel::new("g2_", false);

// Create a composite model (g1 + g2)
let mut composite = add(g1, g2);

// Optionally add a baseline
composite.add_baseline(0.5);

// Use like any other model
let x = Array1::linspace(-10.0, 10.0, 200);
let y = composite.eval(&x).unwrap();

// Fit the composite model to data
let result = fit(&mut composite, x, y_data).unwrap();
```

### Multiplication Example

Applying amplitude modulation to a peak:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, SineModel, multiply};
use ndarray::Array1;

// Create individual models
let mut peak = GaussianModel::new("peak_", false);
let mut modulation = SineModel::new("mod_", false);

// Create a composite model (peak * modulation)
let mut composite = multiply(peak, modulation);

// Use the composite model
let x = Array1::linspace(-10.0, 10.0, 200);
let y = composite.eval(&x).unwrap();
```

## Parameter Handling in Composite Models

When combining models, each component model's parameters are prefixed to ensure uniqueness. For example:

- `GaussianModel::new("g1_", ...)` creates parameters named `g1_amplitude`, `g1_center`, etc.
- `GaussianModel::new("g2_", ...)` creates parameters named `g2_amplitude`, `g2_center`, etc.

This allows you to access and manage parameters specifically for each component:

```rust
// Access individual component parameters
let g1_amplitude = composite.parameters().get("g1_amplitude").unwrap().value();
let g2_center = composite.parameters().get("g2_center").unwrap().value();

// Fix a parameter during fitting
composite.parameters_mut().get_mut("g1_center").unwrap().set_vary(false).unwrap();
```

## Sharing Parameters Between Components

Sometimes you want to constrain parameters across different components of your composite model. For example, you might want two Gaussians to have the same width but different amplitudes and positions.

You can do this using parameter constraints:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, add};
use ndarray::Array1;

// Create individual models
let mut g1 = GaussianModel::new("g1_", false);
let mut g2 = GaussianModel::new("g2_", false);

// Combine them
let mut composite = add(g1, g2);

// Link the sigma (width) parameter between the two Gaussians
composite.parameters_mut().constrain("g2_sigma", "g1_sigma").unwrap();

// Now when g1_sigma changes, g2_sigma will automatically match it
```

## Practical Example: Fitting Multiple Peaks

Here's a complete example of fitting a spectrum with multiple peaks and a baseline:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, add};
use lmopt_rs::uncertainty::uncertainty_analysis;
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data with three peaks
    let x = Array1::linspace(0.0, 10.0, 500);
    
    // Create three Gaussian components with different parameters
    let g1_params = (3.0, 2.0, 0.5); // amplitude, center, sigma
    let g2_params = (4.0, 5.0, 0.7);
    let g3_params = (2.5, 8.0, 0.6);
    
    // Generate data points with noise
    let mut rng = rand::thread_rng();
    let mut y = Array1::zeros(500);
    
    for (i, &x_val) in x.iter().enumerate() {
        // First peak
        let g1 = g1_params.0 * (-(x_val - g1_params.1).powi(2) / (2.0 * g1_params.2.powi(2))).exp();
        
        // Second peak
        let g2 = g2_params.0 * (-(x_val - g2_params.1).powi(2) / (2.0 * g2_params.2.powi(2))).exp();
        
        // Third peak
        let g3 = g3_params.0 * (-(x_val - g3_params.1).powi(2) / (2.0 * g3_params.2.powi(2))).exp();
        
        // Add baseline and noise
        let noise = rng.gen_range(-0.2..0.2);
        y[i] = g1 + g2 + g3 + 0.5 + noise;
    }
    
    // Create individual models
    let mut g1 = GaussianModel::new("g1_", false);
    let mut g2 = GaussianModel::new("g2_", false);
    let mut g3 = GaussianModel::new("g3_", false);
    
    // Give reasonable initial guesses
    g1.parameters_mut().get_mut("g1_amplitude").unwrap().set_value(2.0).unwrap();
    g1.parameters_mut().get_mut("g1_center").unwrap().set_value(2.5).unwrap();
    g1.parameters_mut().get_mut("g1_sigma").unwrap().set_value(0.6).unwrap();
    
    g2.parameters_mut().get_mut("g2_amplitude").unwrap().set_value(3.0).unwrap();
    g2.parameters_mut().get_mut("g2_center").unwrap().set_value(5.5).unwrap();
    g2.parameters_mut().get_mut("g2_sigma").unwrap().set_value(0.8).unwrap();
    
    g3.parameters_mut().get_mut("g3_amplitude").unwrap().set_value(2.0).unwrap();
    g3.parameters_mut().get_mut("g3_center").unwrap().set_value(7.5).unwrap();
    g3.parameters_mut().get_mut("g3_sigma").unwrap().set_value(0.7).unwrap();
    
    // Set bounds to keep centers in reasonable ranges
    g1.parameters_mut().get_mut("g1_center").unwrap().set_bounds(0.0, 4.0).unwrap();
    g2.parameters_mut().get_mut("g2_center").unwrap().set_bounds(3.0, 7.0).unwrap();
    g3.parameters_mut().get_mut("g3_center").unwrap().set_bounds(6.0, 10.0).unwrap();
    
    // Create composite model with baseline
    let mut composite = add(g1, g2);
    composite = add(composite, g3);
    composite.add_baseline(0.5);
    
    // Fit the model to the data
    let fit_result = fit(&mut composite, x.clone(), y.clone())?;
    
    println!("Fit success: {}", fit_result.success);
    println!("Chi-square: {:.6}", fit_result.cost);
    
    // Print fitted parameters
    for param_name in ["g1_amplitude", "g1_center", "g1_sigma",
                       "g2_amplitude", "g2_center", "g2_sigma",
                       "g3_amplitude", "g3_center", "g3_sigma",
                       "baseline"] {
        let value = composite.parameters().get(param_name).unwrap().value();
        let error = fit_result.standard_errors.get(param_name).unwrap_or(&0.0);
        println!("{}: {:.3} ± {:.3}", param_name, value, error);
    }
    
    // Calculate uncertainty
    let uncertainty = uncertainty_analysis(&composite, &fit_result)?;
    
    // Access individual peak areas
    let g1_area = g1_params.0 * g1_params.2 * (2.0 * std::f64::consts::PI).sqrt();
    let g2_area = g2_params.0 * g2_params.2 * (2.0 * std::f64::consts::PI).sqrt();
    let g3_area = g3_params.0 * g3_params.2 * (2.0 * std::f64::consts::PI).sqrt();
    let total_area = g1_area + g2_area + g3_area;
    
    println!("\nTrue peak areas:");
    println!("G1 area: {:.2} ({:.1}%)", g1_area, 100.0 * g1_area / total_area);
    println!("G2 area: {:.2} ({:.1}%)", g2_area, 100.0 * g2_area / total_area);
    println!("G3 area: {:.2} ({:.1}%)", g3_area, 100.0 * g3_area / total_area);
    println!("Total: {:.2}", total_area);
    
    // Calculate fitted peak areas
    let fit_g1_amp = composite.parameters().get("g1_amplitude").unwrap().value();
    let fit_g1_sigma = composite.parameters().get("g1_sigma").unwrap().value();
    let fit_g1_area = fit_g1_amp * fit_g1_sigma * (2.0 * std::f64::consts::PI).sqrt();
    
    let fit_g2_amp = composite.parameters().get("g2_amplitude").unwrap().value();
    let fit_g2_sigma = composite.parameters().get("g2_sigma").unwrap().value();
    let fit_g2_area = fit_g2_amp * fit_g2_sigma * (2.0 * std::f64::consts::PI).sqrt();
    
    let fit_g3_amp = composite.parameters().get("g3_amplitude").unwrap().value();
    let fit_g3_sigma = composite.parameters().get("g3_sigma").unwrap().value();
    let fit_g3_area = fit_g3_amp * fit_g3_sigma * (2.0 * std::f64::consts::PI).sqrt();
    
    let fit_total_area = fit_g1_area + fit_g2_area + fit_g3_area;
    
    println!("\nFitted peak areas:");
    println!("G1 area: {:.2} ({:.1}%)", fit_g1_area, 100.0 * fit_g1_area / fit_total_area);
    println!("G2 area: {:.2} ({:.1}%)", fit_g2_area, 100.0 * fit_g2_area / fit_total_area);
    println!("G3 area: {:.2} ({:.1}%)", fit_g3_area, 100.0 * fit_g3_area / fit_total_area);
    println!("Total: {:.2}", fit_total_area);
    
    Ok(())
}
```

## Advanced Techniques

### 1. Creating Derived Parameters

You can add derived parameters to calculate quantities of interest from model parameters:

```rust
// Calculate peak area from amplitude and sigma
composite.parameters_mut().add_param_with_expr(
    "g1_area", 
    0.0, 
    "g1_amplitude * g1_sigma * sqrt(2*pi)"
).unwrap();
```

### 2. Parameter Constraints Between Components

You can create more complex constraints between components:

```rust
// Set second peak's amplitude to be half of first peak
composite.parameters_mut().constrain("g2_amplitude", "0.5 * g1_amplitude").unwrap();

// Make third peak's width related to its position
composite.parameters_mut().constrain("g3_sigma", "g3_center / 10").unwrap();
```

### 3. Using Different Model Types

You can mix different model types in a composite:

```rust
use lmopt_rs::model::{fit, Model};
use lmopt_rs::models::{GaussianModel, LorentzianModel, StepModel, add};

// Mix Gaussian and Lorentzian peaks
let mut g1 = GaussianModel::new("g1_", false);
let mut l1 = LorentzianModel::new("l1_", false);

// Add a step function
let mut step = StepModel::new("step_", false);

// Combine all of them
let mut composite = add(g1, l1);
composite = add(composite, step);
```

### 4. Progressive Fitting Strategy

For complex fits, you can use a progressive strategy:

```rust
// 1. First fit just the largest peak and baseline
g1.parameters_mut().get_mut("g1_amplitude").unwrap().set_value(3.0).unwrap();
g1.parameters_mut().get_mut("g1_center").unwrap().set_value(5.0).unwrap();
let mut simple = g1.clone();
simple.add_baseline(0.5);
let simple_result = fit(&mut simple, x.clone(), y.clone())?;

// 2. Use those results as starting points for the full model
composite.parameters_mut().get_mut("g1_amplitude").unwrap()
    .set_value(simple.parameters().get("g1_amplitude").unwrap().value()).unwrap();
composite.parameters_mut().get_mut("g1_center").unwrap()
    .set_value(simple.parameters().get("g1_center").unwrap().value()).unwrap();
composite.parameters_mut().get_mut("baseline").unwrap()
    .set_value(simple.parameters().get("baseline").unwrap().value()).unwrap();

// 3. Fix those values initially
composite.parameters_mut().get_mut("g1_amplitude").unwrap().set_vary(false).unwrap();
composite.parameters_mut().get_mut("g1_center").unwrap().set_vary(false).unwrap();
composite.parameters_mut().get_mut("baseline").unwrap().set_vary(false).unwrap();

// 4. Fit just the second and third peaks
let partial_result = fit(&mut composite, x.clone(), y.clone())?;

// 5. Now unfix everything and do final fit
composite.parameters_mut().get_mut("g1_amplitude").unwrap().set_vary(true).unwrap();
composite.parameters_mut().get_mut("g1_center").unwrap().set_vary(true).unwrap();
composite.parameters_mut().get_mut("baseline").unwrap().set_vary(true).unwrap();
let final_result = fit(&mut composite, x.clone(), y.clone())?;
```

## Best Practices for Composite Models

1. **Use Prefixes**: Always use clear, descriptive prefixes for component models
2. **Initial Guesses**: Provide good initial guesses for all components
3. **Set Bounds**: Use bounds to keep parameters in physically reasonable ranges
4. **Progressive Fitting**: For complex models, fit simpler models first and use their results as starting points
5. **Constraints**: Use constraints to enforce relationships between parameters
6. **Model Selection**: Compare different compositions (using AIC or BIC) to find the best model

## Performance Considerations

Composite models can be computationally expensive, especially with many components. Some tips to improve performance:

1. **Limit Components**: Use only as many components as necessary
2. **Fix Parameters**: Fix parameters that are well-determined or not crucial
3. **Use Bounds**: Tight bounds can speed up convergence
4. **Progressive Fitting**: The staged approach can find good solutions faster
5. **Initial Guesses**: Better initial values reduce the number of iterations needed

## Troubleshooting

### Common Issues

1. **Fit doesn't converge**: Try better initial guesses or use bounds
2. **Parameters hit bounds**: The bounds might be too restrictive or the model might not be appropriate
3. **Highly correlated parameters**: Use constraints or a simpler model
4. **Very different parameter magnitudes**: Adjust initial values to be closer to expected final values

### Solutions

1. **Use a staged approach**: Fit simpler models first
2. **Add constraints**: Constrain related parameters
3. **Fix some parameters**: Fix well-determined parameters
4. **Try different initial values**: Experiment with different starting points
5. **Change model composition**: Try different component types or combinations