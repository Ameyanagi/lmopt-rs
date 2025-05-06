# Automatic Differentiation in lmopt-rs

This document describes the automatic differentiation (AD) capabilities in the lmopt-rs library and provides guidance on how to leverage them for optimization problems.

## Overview

Automatic differentiation is a technique for efficiently and accurately computing derivatives of numeric functions. Unlike symbolic differentiation (which can lead to expression swell) or numerical differentiation (which can suffer from truncation and round-off errors), automatic differentiation computes exact derivatives by applying the chain rule to elementary operations.

In lmopt-rs, we use automatic differentiation for:

1. Computing Jacobian matrices for nonlinear least-squares problems
2. Computing gradients for optimization
3. Calculating Hessian matrices when needed for trust-region methods

## Rust's std::autodiff (Nightly Only)

lmopt-rs uses Rust's experimental `std::autodiff` module available in nightly Rust. This module is based on Enzyme, which performs automatic differentiation at the LLVM IR level. The module provides:

- Automatic differentiation capabilities through an attribute macro
- Support for both forward and reverse mode differentiation
- Efficient computation of gradients and Jacobians
- Advanced features for handling complex cases and optimizing performance

The core of this functionality is the `#[autodiff]` attribute macro:

```rust
#[autodiff(NAME, MODE, INPUT_ACTIVITIES..., OUTPUT_ACTIVITY)]
```

Where:
- `NAME`: A valid function name for the generated derivative function
- `MODE`: One of `Forward`, `Reverse`, `ForwardFirst`, `ReverseFirst`, or `Reverse(n)` for batch modes
- `INPUT_ACTIVITIES`: Activity type for each input parameter (Active, Duplicated, Const, DuplicatedNoNeed)
- `OUTPUT_ACTIVITY`: Activity for the output (Active, DuplicatedNoNeed)

### Activity Types

The activity types control how autodiff handles parameters and their derivatives:

- **Active**: The parameter is active in differentiation, and its gradient is returned by value
- **Duplicated**: The parameter is active, and its gradient is accumulated in-place using a mutable reference
- **Const**: The parameter is treated as constant (no gradient needed)
- **DuplicatedNoNeed**: Like Duplicated, but indicates that the original return value isn't needed

### Forward vs Reverse Mode

- **Forward mode**: More efficient for functions with few inputs and many outputs
- **Reverse mode**: More efficient for functions with many inputs and few outputs

For optimization problems, reverse mode is often more appropriate since we typically have many parameters (inputs) and few outputs (scalar objective function or residuals).

### Example Usage: Simple Function

Here's a simple example of computing a gradient for a function f(x,y) = x² + 3y:

```rust
use std::autodiff;

#[autodiff(df, Reverse, Active, Active, Active)]
fn f(x: f32, y: f32) -> f32 {
    x * x + 3.0 * y
}

fn main() {
    let (x, y) = (5.0, 7.0);
    let (z, dx, dy) = df(x, y, 1.0);
    assert_eq!(46.0, z);  // Original function value: 5² + 3*7 = 25 + 21 = 46
    assert_eq!(10.0, dx); // Derivative with respect to x: 2x = 10
    assert_eq!(3.0, dy);  // Derivative with respect to y: 3
}
```

### Example Usage: References and In-place Gradients

For references and pointers, the gradients can be accumulated in-place:

```rust
use std::autodiff;

#[autodiff(df, Reverse, Active, Duplicated, Active)]
fn f(x: f32, y: &f32) -> f32 {
    x * x + 3.0 * y
}

fn main() {
    let x = 5.0;
    let y = 7.0;
    let mut dy = 0.0;
    let (dx, z) = df(x, &y, &mut dy, 1.0);
    assert_eq!(46.0, z);
    assert_eq!(10.0, dx);
    assert_eq!(3.0, dy);
}
```

### Example Usage: Optimizing a Neural Network

For complex use cases like optimizing a neural network, you can optimize performance by specifying which parameters need gradients:

```rust
use std::autodiff;

#[autodiff(backprop, Reverse, Const, Duplicated, DuplicatedNoNeed)]
fn training_loss(images: &[f32], weights: &[f32]) -> f32 {
    // Compute the loss from the images and weights
    let loss = compute_loss(images, weights);
    loss
}

fn train_step(images: &[f32], weights: &mut [f32], learning_rate: f32) {
    let mut gradients = vec![0.0; weights.len()];
    
    // Compute gradients with respect to weights only (images are Const)
    backprop(images, weights, &mut gradients);
    
    // Update weights using gradient descent
    for (w, g) in weights.iter_mut().zip(gradients.iter()) {
        *w -= learning_rate * g;
    }
}
```

## Implementation in lmopt-rs

The lmopt-rs library uses a tiered approach to calculating derivatives:

1. **First tier**: User-provided analytical derivatives (when available)
2. **Second tier**: Automatic differentiation (preferred method)
3. **Third tier**: Numerical differentiation (fallback method)

Our `autodiff` module wraps Rust's experimental autodiff capabilities and provides a consistent interface:

```rust
// Core functions
pub fn jacobian(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>>;
pub fn gradient<F>(f: F, params: &Array1<f64>) -> Result<Array1<f64>> 
    where F: Fn(&Array1<f64>) -> Result<f64>;
pub fn hessian<F>(f: F, params: &Array1<f64>) -> Result<Array2<f64>>
    where F: Fn(&Array1<f64>) -> Result<f64>;
```

## Usage in Problem Implementation

When implementing the `Problem` trait, you can rely on the default implementation of the `jacobian` method, which will use automatic differentiation:

```rust
impl Problem for MyProblem {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        // Implement your residual function
        // ...
    }
    
    fn parameter_count(&self) -> usize {
        // Return number of parameters
    }
    
    fn residual_count(&self) -> usize {
        // Return number of residuals
    }
    
    // The default jacobian implementation will use autodiff
    // No need to implement unless you have analytical derivatives
}
```

If you have analytical derivatives, you can provide them for better performance:

```rust
impl Problem for MyProblemWithAnalyticalDerivatives {
    // ...
    
    fn jacobian(&self, params: &Array1<f64>) -> Result<Array2<f64>> {
        // Your analytical Jacobian implementation
    }
    
    fn has_custom_jacobian(&self) -> bool {
        true
    }
}
```

## Under the Hood: How We Compute the Jacobian

For a Problem with `m` residuals and `n` parameters, computing the Jacobian involves:

1. Creating autodiff wrappers for the residual functions
2. Using the appropriate mode (forward or reverse) based on dimensions:
   - If n < m: Forward mode (differentiate each parameter)
   - If n ≥ m: Reverse mode (differentiate each residual)

### Implementation Detail

```rust
// Pseudocode illustrating how we compute the Jacobian
#[cfg(feature = "nightly_autodiff")]
fn compute_jacobian(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>> {
    let n_params = params.len();
    let n_residuals = problem.residual_count();
    
    // Choose the appropriate mode based on dimensions
    if n_params < n_residuals {
        // Forward mode is more efficient
        compute_jacobian_forward(problem, params)
    } else {
        // Reverse mode is more efficient
        compute_jacobian_reverse(problem, params)
    }
}
```

## Performance Considerations

1. **Analytical vs. Autodiff vs. Numerical**: 
   - Analytical derivatives are typically fastest but require manual derivation
   - Autodiff provides exact derivatives with reasonable performance
   - Numerical differentiation is a last resort due to potential inaccuracies

2. **Forward vs. Reverse Mode**:
   - Forward mode is more efficient when there are few inputs and many outputs
   - Reverse mode is more efficient when there are many inputs and few outputs
   - For Jacobian calculations, we select the appropriate mode based on dimensions

3. **Memory Usage**:
   - Autodiff can require additional memory to store the computational graph
   - For very large problems, this could become a bottleneck

## Limitations and Considerations

1. **Nightly-Only Feature**: The `std::autodiff` module is an experimental feature available only in nightly Rust builds

2. **Non-differentiable Functions**: Autodiff cannot handle non-differentiable functions (e.g., functions with discontinuities)

3. **Complex Control Flow**: Functions with complex control flow (e.g., conditionals that depend on the input values) may cause issues for some autodiff implementations

4. **External Function Calls**: Calls to external functions that the autodiff library cannot analyze will break the chain of differentiation

5. **Performance Overhead**: There is some performance overhead compared to hand-coded derivatives

## Fallback to Numerical Differentiation

When automatic differentiation is not applicable, the library falls back to numerical differentiation:

```rust
// If autodiff is not available or fails
#[cfg(not(feature = "nightly_autodiff"))]
pub fn jacobian(problem: &dyn Problem, params: &Array1<f64>) -> Result<Array2<f64>> {
    finite_difference::jacobian(problem, params, None)
}
```

## Future Improvements

1. **Stabilization of Rust's Autodiff Feature**: 
   As Rust's built-in autodiff stabilizes, we plan to remove the feature flag and make it a standard part of the library.

2. **GPU Acceleration**:
   Future versions may provide GPU-accelerated autodiff for large-scale problems.

3. **Sparsity Exploitation**:
   We plan to add support for exploiting sparsity patterns in Jacobians and Hessians.

## References

- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [Rust std::autodiff documentation](https://doc.rust-lang.org/nightly/std/autodiff/index.html)
- [GitHub issue tracking autodiff implementation](https://github.com/rust-lang/rust/issues/124509)
- [The Chain Rule for Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule)