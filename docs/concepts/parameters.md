# Parameter System

The `lmopt-rs` parameter system provides a flexible way to manage parameters in optimization problems, similar to the parameter handling in lmfit-py. This system allows you to:

- Set bounds (min/max constraints)
- Create algebraic constraints between parameters
- Define parameter expressions
- Fix parameters during optimization
- Organize parameters with prefix grouping

## Basic Parameter Usage

```rust
use lmopt_rs::parameters::{Parameter, Parameters};

// Create a parameter collection
let mut params = Parameters::new();

// Add simple parameters
params.add_param("a", 1.0)?;  // name and initial value

// Add parameters with bounds
params.add_param_with_bounds("amplitude", 2.0, 0.0, 10.0)?;  // min=0, max=10

// Fix a parameter (won't be varied during optimization)
params.get_mut("a").unwrap().set_vary(false)?;

// Access parameter values
let a_value = params.get("a").unwrap().value();
let amplitude = params.get("amplitude").unwrap().value();

// Update parameter values
params.get_mut("a").unwrap().set_value(2.5)?;
```

## Parameter Bounds

Bounds allow you to constrain parameters within specific ranges:

```rust
// Parameter with lower bound only
params.add_param_with_bounds("decay", 1.0, 0.0, f64::INFINITY)?;

// Parameter with upper bound only
params.add_param_with_bounds("phase", 0.0, f64::NEG_INFINITY, 2.0 * std::f64::consts::PI)?;

// Parameter with both bounds
params.add_param_with_bounds("scale", 1.0, 0.1, 100.0)?;
```

During optimization, parameters will always stay within their bounds. This is implemented using parameter transformations that convert bounded parameters to unbounded space for the optimizer.

## Parameter Expressions

Parameter expressions allow you to define parameters that are calculated from other parameters:

```rust
// Add a parameter with an expression
params.add_param_with_expr("fwhm", 2.355, "2.355 * sigma")?;

// More complex expressions
params.add_param_with_expr("area", 0.0, "amplitude * sigma * sqrt(2*pi)")?;
params.add_param_with_expr("height", 0.0, "amplitude / (sigma * sqrt(2*pi))")?;

// Parameters in expressions are updated automatically
params.get_mut("sigma").unwrap().set_value(1.5)?;
let fwhm = params.get("fwhm").unwrap().value();  // now equals 2.355 * 1.5
```

Expressions are parsed and evaluated during the optimization process. When a parameter used in an expression changes, all dependent parameters are automatically updated.

## Parameter Constraints

Constraints create relationships between parameters:

```rust
// Constrain one parameter to equal another parameter
params.constrain("width1", "width2")?;

// Constrain a parameter to a more complex expression
params.constrain("area", "pi * radius^2")?;

// Constraints with multiplicative factors
params.constrain("sigma2", "2 * sigma1")?;
```

When a parameter is constrained, it becomes dependent on other parameters and will not be varied directly during optimization.

## Parameter Prefixes

Prefixes help organize parameters for multiple components:

```rust
// Create models with different prefixes
let mut model1 = GaussianModel::new("g1_", true);
let mut model2 = GaussianModel::new("g2_", true);

// Access parameters by their prefixed names
model1.parameters().get("g1_amplitude").unwrap().value();
model2.parameters().get("g2_center").unwrap().value();
```

This is especially useful when working with composite models where multiple components might have parameters with the same base name.

## Converting Between Parameter Collections and Arrays

For optimization algorithms that work with parameter arrays:

```rust
// Convert Parameters to an Array1 for the optimizer
let param_array = params.to_array()?;

// After optimization, update Parameters from the optimized array
params.update_from_array(&optimized_params)?;
```

## Using ParameterProblem Trait

The `ParameterProblem` trait simplifies working with parameter-based optimization problems:

```rust
use lmopt_rs::{LmOptError, Result};
use lmopt_rs::parameters::Parameters;
use lmopt_rs::problem_params::ParameterProblem;
use ndarray::Array1;

struct MyModel {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
    parameters: Parameters,
}

impl ParameterProblem for MyModel {
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
        
        // Calculate residuals using current parameter values
        let residuals = self.x_data.iter()
            .zip(self.y_data.iter())
            .map(|(&x, &y)| a * x + b - y)
            .collect::<Vec<f64>>();
            
        Ok(Array1::from_vec(residuals))
    }
    
    fn residual_count(&self) -> usize {
        self.x_data.len()
    }
}
```

This trait makes it easy to use the parameter system with the optimization algorithms, and provides automatic conversion to the Problem trait.

## Advanced Usage

### Parameter Initialization Hints

```rust
// Set hint for auto-initialization
params.get_mut("amplitude").unwrap().set_hint("peak_height")?;
params.get_mut("center").unwrap().set_hint("peak_position")?;
```

### User Info

```rust
// Store additional information with parameters
params.get_mut("temperature").unwrap().set_user_info("Measured at room temperature")?;
```

### Custom Parameter Transformations

For specialized bounded parameters:

```rust
params.add_param_with_transform("angle", 0.0, Box::new(MyCustomTransform))?;
```

Where `MyCustomTransform` is a custom implementation of the `ParameterTransform` trait.

### Parameter Serialization

```rust
// Convert parameters to JSON
let json = serde_json::to_string(&params)?;

// Load parameters from JSON
let loaded_params: Parameters = serde_json::from_str(&json)?;
```