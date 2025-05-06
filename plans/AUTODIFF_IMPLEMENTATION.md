# Automatic Differentiation Implementation Plan

This document details the plan for implementing automatic differentiation support in the lmopt-rs library, focusing on leveraging Rust's experimental std::autodiff module when available and providing effective fallbacks when it's not.

## Overview

Automatic differentiation (AD) is a key component of our library, enabling accurate and efficient computation of derivatives for optimization problems. We'll use a tiered approach, prioritizing user-provided analytical derivatives, then autodiff, and finally falling back to numerical differentiation when necessary.

## Implementation Strategy

### 1. Feature-gated Implementation

We'll use Rust's feature gates to conditionally include autodiff functionality:

```rust
// In Cargo.toml
[features]
default = []
nightly_autodiff = []  # Feature flag for std::autodiff functionality (requires nightly Rust)
```

This allows the library to work on both stable and nightly Rust, with enhanced functionality on nightly.

### 2. Layered API Design

Our autodiff module will provide a consistent API regardless of whether the underlying std::autodiff feature is available:

```rust
// Public API (always available)
pub fn jacobian(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>>;
pub fn gradient<F>(f: F, params: &Array1<f64>) -> Result<Array1<f64>>
    where F: Fn(&Array1<f64>) -> Result<f64>;
pub fn hessian<F>(f: F, params: &Array1<f64>) -> Result<Array2<f64>>
    where F: Fn(&Array1<f64>) -> Result<f64>;
```

Internally, these functions will dispatch to the appropriate implementation:

```rust
// Implementation
pub fn jacobian(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>> {
    // Check for user-provided analytical Jacobian
    if problem.has_custom_jacobian() {
        return problem.jacobian(params);
    }
    
    // Try autodiff when available
    #[cfg(feature = "nightly_autodiff")]
    {
        match jacobian_autodiff(problem, params) {
            Ok(jac) => return Ok(jac),
            Err(_) => {}  // Fall through to numerical differentiation
        }
    }
    
    // Fall back to numerical differentiation
    finite_difference::jacobian(problem, params, None)
}
```

### 3. Autodiff Wrapper Implementation

When the nightly_autodiff feature is enabled, we'll implement wrappers around std::autodiff:

```rust
#[cfg(feature = "nightly_autodiff")]
fn jacobian_autodiff(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>> {
    let n_params = params.len();
    let n_residuals = problem.residual_count();
    let mut jac = Array2::zeros((n_residuals, n_params));
    
    // Choose the most efficient autodiff mode based on dimensions
    if n_params < n_residuals {
        // Forward mode is more efficient (fewer parameters than residuals)
        jacobian_forward_mode(problem, params, &mut jac)
    } else {
        // Reverse mode is more efficient (more parameters than residuals)
        jacobian_reverse_mode(problem, params, &mut jac)
    }
}
```

### 4. Forward-Mode Implementation

For problems with fewer parameters than residuals, forward-mode autodiff is more efficient:

```rust
#[cfg(feature = "nightly_autodiff")]
fn jacobian_forward_mode(problem: &dyn Problem, params: &Array1<f64>, jac: &mut Array2<f64>) -> Result<Array2<f64>> {
    let n_params = params.len();
    let n_residuals = problem.residual_count();
    let params_vec = params.to_vec();
    
    // Create a wrapper function that converts from slice to Array1 and back
    fn wrapper_fn(problem: &dyn Problem, p: &[f64]) -> Vec<f64> {
        let params_array = Array1::from_slice(p);
        match problem.eval(&params_array) {
            Ok(residuals) => residuals.to_vec(),
            Err(_) => vec![0.0; problem.residual_count()],
        }
    }
    
    // Use the autodiff attribute macro to generate the Jacobian function
    // This code will need to be adapted based on the exact API of std::autodiff
    #[autodiff(df_dx, Forward, [Active; n_params], [Active; n_residuals])]
    fn wrapped_eval(p: &[f64]) -> Vec<f64> {
        wrapper_fn(problem, p)
    }
    
    // For each parameter, compute its derivatives
    for i in 0..n_params {
        // Create seed vector with 1.0 at position i
        let mut seed = vec![0.0; n_params];
        seed[i] = 1.0;
        
        // Compute derivatives using forward mode
        let derivatives = df_dx(&params_vec, &seed);
        
        // Fill column i of the Jacobian
        for j in 0..n_residuals {
            jac[[j, i]] = derivatives[j];
        }
    }
    
    Ok(jac.clone())
}
```

### 5. Reverse-Mode Implementation

For problems with more parameters than residuals, reverse-mode autodiff is more efficient:

```rust
#[cfg(feature = "nightly_autodiff")]
fn jacobian_reverse_mode(problem: &dyn Problem, params: &Array1<f64>, jac: &mut Array2<f64>) -> Result<Array2<f64>> {
    let n_params = params.len();
    let n_residuals = problem.residual_count();
    let params_vec = params.to_vec();
    
    // For each residual, compute its gradient with respect to all parameters
    for i in 0..n_residuals {
        // Create a wrapper function that returns just the i-th residual
        fn residual_i_fn(problem: &dyn Problem, i: usize, p: &[f64]) -> f64 {
            let params_array = Array1::from_slice(p);
            match problem.eval(&params_array) {
                Ok(residuals) => {
                    if i < residuals.len() {
                        residuals[i]
                    } else {
                        0.0
                    }
                },
                Err(_) => 0.0,
            }
        }
        
        // Use the autodiff attribute macro for the i-th residual
        #[autodiff(df_i, Reverse, [Active; n_params], Active)]
        fn wrapped_residual_i(p: &[f64]) -> f64 {
            residual_i_fn(problem, i, p)
        }
        
        // Compute derivatives using reverse mode
        let (_, derivatives) = df_i(&params_vec, 1.0);
        
        // Fill row i of the Jacobian
        for j in 0..n_params {
            jac[[i, j]] = derivatives[j];
        }
    }
    
    Ok(jac.clone())
}
```

### 6. Gradient Implementation

For scalar functions, the gradient calculation is simplified:

```rust
#[cfg(feature = "nightly_autodiff")]
fn gradient_autodiff<F>(f: &F, params: &Array1<f64>) -> Result<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64>
{
    let n_params = params.len();
    let params_vec = params.to_vec();
    
    // Create a wrapper function that converts from slice to Array1
    fn wrapper_fn<F>(f: &F, p: &[f64]) -> f64
    where
        F: Fn(&Array1<f64>) -> Result<f64>
    {
        let params_array = Array1::from_slice(p);
        match f(&params_array) {
            Ok(value) => value,
            Err(_) => f64::INFINITY,  // Use infinity for error cases
        }
    }
    
    // Use the autodiff attribute macro to generate the gradient function
    #[autodiff(grad_f, Reverse, [Active; n_params], Active)]
    fn wrapped_f(p: &[f64]) -> f64 {
        wrapper_fn(f, p)
    }
    
    // Compute the gradient using reverse mode
    let (_, gradient_vec) = grad_f(&params_vec, 1.0);
    
    // Convert to Array1
    let gradient = Array1::from_vec(gradient_vec);
    
    Ok(gradient)
}
```

### 7. Hessian Implementation

For the Hessian matrix (second derivatives), we have two options:

1. Use direct second-order autodiff if supported
2. Apply autodiff twice (to the gradient function)

```rust
#[cfg(feature = "nightly_autodiff")]
fn hessian_autodiff<F>(f: &F, params: &Array1<f64>) -> Result<Array2<f64>>
where
    F: Fn(&Array1<f64>) -> Result<f64>
{
    let n_params = params.len();
    let mut hess = Array2::zeros((n_params, n_params));
    let params_vec = params.to_vec();
    
    // First, compute the gradient function
    fn wrapper_fn<F>(f: &F, p: &[f64]) -> f64
    where
        F: Fn(&Array1<f64>) -> Result<f64>
    {
        let params_array = Array1::from_slice(p);
        match f(&params_array) {
            Ok(value) => value,
            Err(_) => f64::INFINITY,
        }
    }
    
    // Use autodiff to get the gradient function
    #[autodiff(grad_f, Reverse, [Active; n_params], Active)]
    fn wrapped_f(p: &[f64]) -> f64 {
        wrapper_fn(f, p)
    }
    
    // Apply autodiff again to each component of the gradient
    for i in 0..n_params {
        // Create a function that returns the i-th component of the gradient
        fn grad_i_fn<F>(f: &F, i: usize, p: &[f64]) -> f64
        where
            F: Fn(&Array1<f64>) -> Result<f64>
        {
            let (_, grad) = grad_f(p, 1.0);
            grad[i]
        }
        
        // Use autodiff to compute the gradient of the i-th gradient component
        #[autodiff(hess_i, Reverse, [Active; n_params], Active)]
        fn wrapped_grad_i(p: &[f64]) -> f64 {
            grad_i_fn(f, i, p)
        }
        
        // Compute the i-th row of the Hessian
        let (_, hess_row) = hess_i(&params_vec, 1.0);
        
        // Fill the i-th row of the Hessian
        for j in 0..n_params {
            hess[[i, j]] = hess_row[j];
        }
    }
    
    Ok(hess)
}
```

## Numerical Differentiation Fallback

When autodiff is not available or fails, we'll fall back to numerical differentiation:

```rust
// Fallback implementation
pub fn jacobian(problem: &dyn Problem, params: &Array1<f64>, epsilon: Option<f64>) -> Result<Array2<f64>> {
    let eps = epsilon.unwrap_or(DEFAULT_EPSILON);
    let n_params = params.len();
    let n_residuals = problem.residual_count();
    
    // Evaluate residuals at the initial point
    let residuals = problem.eval(params)?;
    
    // Initialize Jacobian matrix
    let mut jac = Array2::zeros((n_residuals, n_params));
    
    // Compute Jacobian using forward differences
    for j in 0..n_params {
        // Perturb j-th parameter
        let mut params_perturbed = params.clone();
        params_perturbed[j] += eps;
        
        // Evaluate residuals at perturbed point
        let residuals_perturbed = problem.eval(&params_perturbed)?;
        
        // Compute partial derivatives
        for i in 0..n_residuals {
            jac[[i, j]] = (residuals_perturbed[i] - residuals[i]) / eps;
        }
    }
    
    Ok(jac)
}
```

## Integration with Problem Trait

The autodiff functionality will be integrated with our Problem trait:

```rust
pub trait Problem {
    // Other methods...
    
    /// Evaluate the Jacobian matrix.
    ///
    /// The default implementation uses automatic differentiation when available,
    /// falling back to numerical differentiation when necessary.
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        crate::utils::autodiff::jacobian(self, params)
    }
    
    /// Check if this problem provides a custom Jacobian implementation.
    fn has_custom_jacobian(&self) -> bool {
        false
    }
}
```

## Examples

We'll provide examples demonstrating how to use autodiff in different scenarios:

1. **Basic Example**: Simple function differentiation
2. **Problem Example**: Using autodiff with the Problem trait
3. **Custom Jacobian Example**: Providing analytical derivatives for comparison
4. **Performance Comparison**: Benchmarking autodiff vs. numerical differentiation

## Testing Strategy

We'll create comprehensive tests for the autodiff functionality:

1. **Unit Tests**: Test individual autodiff functions
2. **Comparison Tests**: Compare autodiff results with analytical derivatives
3. **Edge Case Tests**: Test with tricky functions (e.g., highly nonlinear)
4. **Performance Tests**: Benchmark autodiff performance
5. **Feature Gate Tests**: Ensure the code works correctly with and without the nightly_autodiff feature

## Timeline

| Task | Estimated Duration | Dependencies |
|------|-------------------|--------------|
| Setup Feature Gates | 1 day | None |
| Implement API Structure | 2 days | Feature Gates |
| Forward Mode Implementation | 3 days | API Structure |
| Reverse Mode Implementation | 3 days | API Structure |
| Gradient Implementation | 2 days | Forward/Reverse Mode |
| Hessian Implementation | 3 days | Gradient Implementation |
| Numerical Fallbacks | 2 days | None |
| Integration Tests | 3 days | All Implementation |
| Examples | 2 days | All Implementation |
| Documentation | 2 days | All Implementation |

## Success Criteria

1. Autodiff produces accurate derivatives comparable to analytical solutions
2. Performance is better than numerical differentiation
3. Feature gates work correctly (library works on both stable and nightly Rust)
4. Code is well-documented and tested
5. Examples demonstrate effective usage patterns