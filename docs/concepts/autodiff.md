# Automatic Differentiation

This document describes the automatic differentiation capabilities in `lmopt-rs`, which allow for exact, efficient computation of derivatives without manual coding or numerical approximation.

## Overview

Automatic differentiation (AD) is a technique for computing derivatives of functions exactly, without using numerical approximations like finite differences. This results in more accurate and typically faster derivative calculations, which are critical for the Levenberg-Marquardt algorithm and other optimization methods.

`lmopt-rs` leverages Rust's experimental autodiff features to provide derivative computations for the Jacobian matrix in a nonlinear least-squares problem.

## Benefits of Automatic Differentiation

Compared to other methods for computing derivatives, autodiff offers several advantages:

1. **Accuracy**: Computes exact derivatives to machine precision, unlike finite differences which introduce approximation errors
2. **Efficiency**: Often faster than numerical differentiation, especially for complex functions
3. **Convenience**: No need to manually code derivatives for each function
4. **Robustness**: Handles complex expressions without differentiation errors

## Implementation in lmopt-rs

In `lmopt-rs`, automatic differentiation is implemented using the experimental `std::autodiff` module in Rust nightly. This module, powered by [Enzyme](https://enzyme.mit.edu/), performs automatic differentiation at the LLVM IR level.

### Enabling Autodiff Feature

To use automatic differentiation features, you need to:

1. Use Rust nightly (already required for `lmopt-rs`)
2. Enable the `autodiff` feature in your `Cargo.toml`:

```toml
[dependencies]
lmopt-rs = { version = "0.1.0", features = ["autodiff"] }
```

### Usage in Problem Definition

When defining a problem, you can specify that the Jacobian should be computed using automatic differentiation:

```rust
use lmopt_rs::{LevenbergMarquardt, Problem, lm::DiffMethod};
use ndarray::{array, Array1};

struct QuadraticProblem;

impl Problem for QuadraticProblem {
    fn eval(&self, params: &Array1<f64>) -> lmopt_rs::Result<Array1<f64>> {
        // Residual calculation as before
        // ...
    }
    
    fn parameter_count(&self) -> usize {
        3
    }
    
    fn residual_count(&self) -> usize {
        3
    }
    
    // Automatic differentiation will be used, no need to implement jacobian()
}

fn main() -> lmopt_rs::Result<()> {
    let problem = QuadraticProblem;
    
    // Configure the optimizer to use autodiff
    let mut optimizer = LevenbergMarquardt::new()
        .with_differentiation_method(DiffMethod::AutoDiff);
    
    // Rest of the code remains the same
    // ...
    
    Ok(())
}
```

When you use `DiffMethod::AutoDiff`, the Jacobian matrix will be computed using automatic differentiation rather than finite differences.

## The #[autodiff] Attribute

Under the hood, `lmopt-rs` uses the `#[autodiff]` attribute macro from Rust's experimental autodiff API. This macro has the following syntax:

```rust
#[autodiff(NAME, MODE, INPUT_ACTIVITIES..., OUTPUT_ACTIVITY)]
```

Where:
- `NAME`: A valid function name for the generated derivative function
- `MODE`: One of `Forward`, `Reverse`, `ForwardFirst`, `ReverseFirst`, or `Reverse(n)` for batch modes
- `INPUT_ACTIVITIES`: Activity type for each input parameter:
  - **Active**: Parameter is active in differentiation, gradient returned by value
  - **Duplicated**: Parameter is active, gradient accumulated in-place via mutable reference
  - **Const**: Parameter treated as constant (no gradient needed)
  - **DuplicatedNoNeed**: Like Duplicated, but original return value isn't needed
- `OUTPUT_ACTIVITY`: Activity for the output (Active or DuplicatedNoNeed)

### Example

Here's a basic example of a function with its derivative using the `#[autodiff]` attribute:

```rust
#[autodiff(df, Reverse, Active, Active, Active)]
fn f(x: f32, y: f32) -> f32 {
    x * x + 3.0 * y
}

// The generated df function returns (original_result, d_dx, d_dy)
// df(5.0, 7.0, 1.0) returns (46.0, 10.0, 3.0)
```

## Forward vs. Reverse Mode

Automatic differentiation has two primary modes:

1. **Forward Mode**: Efficient when there are few inputs and many outputs
2. **Reverse Mode**: Efficient when there are many inputs and few outputs

In the context of Levenberg-Marquardt optimization, we typically have:
- More parameters than residuals: Reverse mode is preferred
- More residuals than parameters: Forward mode is preferred

`lmopt-rs` automatically selects the appropriate mode based on the problem dimensions, optimizing performance.

## Fallback to Numerical Differentiation

If autodiff can't be applied to a particular function (e.g., due to unsupported operations), `lmopt-rs` will automatically fall back to numerical differentiation. This ensures that optimization can continue even when autodiff isn't applicable.

## Performance Considerations

While automatic differentiation is generally faster than numerical differentiation, there are some considerations to keep in mind:

1. **Compilation Time**: Code using autodiff may take longer to compile
2. **Memory Usage**: Reverse mode autodiff requires storing intermediate values
3. **Function Complexity**: Very complex functions might have higher overhead

For the best performance:

- Keep your residual functions as simple as possible
- Avoid unnecessary branching or complex control flow
- Consider pre-computing constants outside the autodiff region

## Limitations

The current implementation has some limitations:

1. Requires Rust nightly
2. Not all Rust operations are supported for autodiff
3. External functions (e.g., from other crates) can't be differentiated unless they also use autodiff
4. Error handling within differentiated functions is limited

## Future Work

In future versions of `lmopt-rs`, we plan to:

1. Expand autodiff support as Rust's autodiff capabilities mature
2. Add more pre-differentiated models and functions
3. Explore ways to make autodiff more robust with complex code

## References

- [Enzyme: High-Performance Automatic Differentiation of LLVM](https://enzyme.mit.edu/)
- [Rust Autodiff RFC](https://github.com/rust-lang/rfcs/pull/3453)
- [Introduction to Automatic Differentiation](https://arxiv.org/abs/1904.10996)