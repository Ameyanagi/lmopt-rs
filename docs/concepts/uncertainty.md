# Uncertainty Analysis

One of the most powerful features of `lmopt-rs` is its comprehensive uncertainty analysis capabilities. Understanding the uncertainty in fitted parameters is crucial for making valid scientific and engineering conclusions.

## Core Uncertainty Concepts

Uncertainty analysis in `lmopt-rs` includes:

1. **Covariance Matrix**: Estimates the uncertainty and correlation between parameters
2. **Standard Errors**: The standard deviation of parameter estimates
3. **Confidence Intervals**: The range within which parameters are likely to lie
4. **Monte Carlo Methods**: Simulation-based approaches for more robust uncertainty estimation

## Basic Uncertainty Analysis

After fitting a model, you can easily calculate uncertainty estimates:

```rust
use lmopt_rs::{
    model::{fit, Model},
    uncertainty::{uncertainty_analysis, covariance_matrix, standard_errors}
};

// Fit a model
let result = fit(&mut model, x_data, y_data)?;

// Basic uncertainty analysis
let uncertainty = uncertainty_analysis(&model, &result)?;

// Print parameter estimates with uncertainties
for (name, interval) in &uncertainty.confidence_intervals {
    let value = model.parameters().get(name).unwrap().value();
    println!("{}: {:.4} ± {:.4} (95% CI: [{:.4}, {:.4}])",
             name, value, interval.error, interval.lower, interval.upper);
}
```

## Covariance Matrix

The covariance matrix is the foundation of parameter uncertainty estimation:

```rust
use lmopt_rs::uncertainty::covariance_matrix;

// Calculate the covariance matrix
let cov_matrix = covariance_matrix(&model, &result)?;

// Access elements
let variance_a = cov_matrix[(0, 0)];  // Variance of first parameter
let covariance_ab = cov_matrix[(0, 1)];  // Covariance between first and second parameters

// Calculate correlation coefficient
let correlation_ab = covariance_ab / (variance_a.sqrt() * cov_matrix[(1, 1)].sqrt());
```

## Standard Errors

Standard errors provide a simple measure of parameter uncertainty:

```rust
use lmopt_rs::uncertainty::standard_errors;

// Calculate standard errors from the covariance matrix
let std_errors = standard_errors(&model, &result)?;

// Or access them from the uncertainty analysis
let uncertainty = uncertainty_analysis(&model, &result)?;
for (name, error) in &uncertainty.standard_errors {
    println!("{}: {:.4} ± {:.4}", name, 
             model.parameters().get(name).unwrap().value(), 
             error);
}
```

## Confidence Intervals

Confidence intervals provide a range estimate for parameters:

```rust
// Calculate 95% confidence intervals
let uncertainty = uncertainty_analysis(&model, &result)?;

for (name, interval) in &uncertainty.confidence_intervals {
    println!("{}: 95% CI [{:.4}, {:.4}]", 
             name, interval.lower, interval.upper);
}

// Access a specific confidence interval
if let Some(interval) = uncertainty.confidence_intervals.get("amplitude") {
    println!("Amplitude: {:.4} ± {:.4}", interval.value, interval.error);
}
```

## Monte Carlo Methods

For more robust uncertainty estimation, especially for nonlinear models:

```rust
use lmopt_rs::uncertainty::{
    monte_carlo_refit, 
    uncertainty_analysis_with_monte_carlo
};

// Perform Monte Carlo analysis with 1000 simulations
let monte_carlo = uncertainty_analysis_with_monte_carlo(&model, &result, 1000)?;

// Access Monte Carlo parameter distributions
for (name, distribution) in &monte_carlo.parameter_distributions {
    let mean = distribution.mean();
    let std_dev = distribution.std_dev();
    let median = distribution.percentile(50.0);
    let p5 = distribution.percentile(5.0);
    let p95 = distribution.percentile(95.0);
    
    println!("{}: mean={:.4}, median={:.4}, std_dev={:.4}", name, mean, median, std_dev);
    println!("   90% CI: [{:.4}, {:.4}]", p5, p95);
}
```

## Error Propagation

Propagate uncertainty to derived quantities:

```rust
use lmopt_rs::uncertainty::propagate_uncertainty;

// Define a function that uses the parameters
let derived_quantity = |params: &Parameters| {
    let a = params.get("a").unwrap().value();
    let b = params.get("b").unwrap().value();
    
    // Calculate something derived from parameters
    a * b.powi(2)
};

// Propagate uncertainty to this derived quantity
let uncertainty = propagate_uncertainty(&model, &result, &derived_quantity)?;

println!("Derived quantity: {:.4} ± {:.4}", uncertainty.value, uncertainty.error);
```

## Uncertainty Calculator

For more control over uncertainty analysis:

```rust
use lmopt_rs::uncertainty::UncertaintyCalculator;

// Create a customized uncertainty calculator
let calculator = UncertaintyCalculator::new()
    .with_confidence_level(0.99)  // 99% confidence intervals
    .with_monte_carlo_samples(2000)  // Use 2000 MC samples
    .with_custom_noise_model(|residual, _| {
        // Custom noise model: residual-dependent noise
        residual.abs().max(0.1)
    });

// Run the analysis
let uncertainty = calculator.analyze(&model, &result)?;
```

## Visualizing Uncertainty

```rust
// Generate confidence bands for plotting
let x_plot = Array1::linspace(x_min, x_max, 200);
let y_fit = model.eval(&x_plot)?;

// Calculate confidence bands
let bands = uncertainty.confidence_bands(&model, &x_plot, 0.95)?;

// bands.lower and bands.upper contain the lower and upper confidence bands
// These can be plotted alongside the fit curve
```

## Advanced Topics

### Covariance Matrix Estimation Methods

```rust
use lmopt_rs::uncertainty::{CovarianceMethod, UncertaintyCalculator};

// Choose a specific method for covariance estimation
let calculator = UncertaintyCalculator::new()
    .with_covariance_method(CovarianceMethod::LaplaceApproximation);
    
// Available methods:
// - Jacobian: Standard method using the Jacobian matrix
// - Hessian: Direct Hessian estimation (more accurate but slower)
// - LaplaceApproximation: Approximation of the Hessian
// - Bootstrap: Resampling-based method
```

### Robust Uncertainty Estimation

For data with outliers or non-Gaussian errors:

```rust
// Use bootstrap method (more robust to outliers)
let calculator = UncertaintyCalculator::new()
    .with_covariance_method(CovarianceMethod::Bootstrap)
    .with_bootstrap_samples(1000);
    
// Or use Monte Carlo with custom error distribution
let calculator = UncertaintyCalculator::new()
    .with_monte_carlo_samples(1000)
    .with_error_distribution(ErrorDistribution::Students { df: 3.0 });
```

### Profile Likelihood Method

For highly nonlinear models where confidence intervals may be asymmetric:

```rust
use lmopt_rs::uncertainty::profile_likelihood;

// Calculate confidence intervals using profile likelihood
let profile = profile_likelihood(&model, "amplitude", &result, 0.95)?;

println!("Profile likelihood 95% CI: [{:.4}, {:.4}]", profile.lower, profile.upper);
```

### Parameter Correlation Visualization

```rust
// Extract correlation matrix
let correlation = uncertainty.correlation_matrix();

// Display parameter correlations
for i in 0..correlation.nrows() {
    for j in 0..correlation.ncols() {
        if i != j {
            let param_i = &model.parameter_names()[i];
            let param_j = &model.parameter_names()[j];
            println!("Correlation between {} and {}: {:.4}", 
                     param_i, param_j, correlation[(i, j)]);
        }
    }
}
```

## Best Practices

1. **Always examine uncertainties**: Never report fitted parameters without their uncertainties.

2. **Check parameter correlations**: High correlations (>0.9) indicate that parameters are not independent and the model might be overparameterized.

3. **Use Monte Carlo for nonlinear models**: Standard covariance-based uncertainty can be misleading for highly nonlinear models.

4. **Consider multiple methods**: Compare uncertainties from different methods to ensure robustness.

5. **Visualize confidence bands**: Plot model predictions with confidence bands to visualize the fit quality.

6. **Watch for hitting bounds**: Parameters at their bounds may have underestimated uncertainties.