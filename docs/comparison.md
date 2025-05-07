# Comparison with Other Libraries

This document compares `lmopt-rs` with similar libraries to help you understand how it relates to existing solutions and why you might choose it for your projects.

## Related Documentation
- [Getting Started Guide](./getting_started.md)
- [Parameter System](./concepts/parameters.md)
- [Model System](./concepts/models.md)
- [Levenberg-Marquardt Algorithm](./concepts/lm_algorithm.md)
- [Uncertainty Analysis](./concepts/uncertainty.md)
- [Global Optimization](./concepts/global_optimization.md)

## Comparison with levenberg-marquardt (Rust)

The [levenberg-marquardt](https://crates.io/crates/levenberg-marquardt) crate provides a Rust implementation of the Levenberg-Marquardt algorithm. While `lmopt-rs` maintains compatibility with this crate, it significantly extends the functionality.

### Key Differences

| Feature | levenberg-marquardt | lmopt-rs |
|---------|---------------------|----------|
| **Parameter System** | Basic parameter arrays | Named parameters with bounds and constraints |
| **Model System** | No built-in model system | Comprehensive model system with many built-in models |
| **Uncertainty Analysis** | None | Full uncertainty quantification with multiple methods |
| **Global Optimization** | None | Multiple global optimization algorithms |
| **Matrix Support** | Limited to nalgebra | Supports nalgebra, ndarray, and faer |
| **Autodiff Support** | None | Experimental support for automatic differentiation |

### API Compatibility

`lmopt-rs` provides a compatibility layer with the `levenberg-marquardt` crate, allowing you to use existing code that implements the `LeastSquaresProblem` trait:

```rust
use lmopt_rs::lm_compat::LmProblemAdapter;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};

// Your existing problem implementation
let problem: MyLmProblem = /* ... */;

// Using with levenberg-marquardt directly
let lm = LevenbergMarquardt::new();
let (_solution, _errors) = lm.minimize(&problem);

// Using with lmopt-rs
let adapter = LmProblemAdapter::new(problem);
let optimizer = lmopt_rs::LevenbergMarquardt::with_default_config();
let result = optimizer.minimize(&adapter, initial_params).unwrap();
```

## Comparison with lmfit-py (Python)

[lmfit-py](https://lmfit.github.io/lmfit-py/) is a popular Python library that extends the capabilities of scipy.optimize by adding parameter handling, constraints, bounds, and uncertainty analysis. `lmopt-rs` brings similar functionality to the Rust ecosystem.

### Key Similarities

- Named parameters with bounds and constraints
- Parameter expressions and derived parameters
- Built-in model system with common models (Gaussian, Lorentzian, etc.)
- Composite models for combining multiple model functions
- Uncertainty analysis with covariance matrix and Monte Carlo methods
- Global optimization capabilities

### Key Differences

| Feature | lmfit-py | lmopt-rs |
|---------|----------|----------|
| **Implementation Language** | Python | Rust |
| **Performance** | Python speed with NumPy | Native Rust performance |
| **Memory Safety** | Python's safety model | Rust's compile-time safety |
| **Concurrency** | Limited by Python's GIL | Full Rust concurrency support |
| **Dependencies** | Requires SciPy, NumPy, etc. | Pure Rust implementation |
| **Interactive Usage** | Strong REPL and notebook support | Primarily library-based usage |
| **Maturity** | Mature, widely used | Newer, actively developed |

### Feature Matrix

| Feature | lmfit-py | lmopt-rs |
|---------|:--------:|:--------:|
| Named Parameters | ✅ | ✅ |
| Parameter Bounds | ✅ | ✅ |
| Parameter Constraints | ✅ | ✅ |
| Parameter Expressions | ✅ | ✅ |
| Built-in Models | ✅ | ✅ |
| Composite Models | ✅ | ✅ |
| Confidence Intervals | ✅ | ✅ |
| Monte Carlo Analysis | ✅ | ✅ |
| Global Optimization | ✅ | ✅ |
| Automatic Differentiation | ❌ | ✅ (Experimental) |
| Serialization | ✅ | ✅ |
| Command-line Interface | ❌ | ❌ |

## When to Choose lmopt-rs

`lmopt-rs` is an excellent choice when:

1. **You need Rust's performance benefits**: When working with large datasets or computationally intensive models, the native Rust implementation can provide significant performance advantages.

2. **Memory safety is critical**: Rust's compile-time safety guarantees can prevent many common programming errors, especially in complex numerical code.

3. **Integration with Rust ecosystem**: For projects already in Rust, `lmopt-rs` provides a seamless integration without requiring Python interop.

4. **Comprehensive optimization needs**: When you need both local and global optimization, uncertainty analysis, and a flexible parameter system in a single library.

5. **Real-time applications**: For applications where predictable performance and low latency are required.

## When to Choose Alternatives

Consider alternatives when:

1. **Python ecosystem integration**: If your project is primarily in Python and deeply integrated with SciPy and NumPy, lmfit-py may be more convenient.

2. **Simpler needs**: If you only need the basic Levenberg-Marquardt algorithm without parameter handling or uncertainty analysis, the simpler `levenberg-marquardt` crate may be sufficient.

3. **Interactive analysis**: For exploratory data analysis in notebooks or REPL environments, Python-based solutions offer more interactivity.

4. **Established workflows**: If you or your team already have established workflows with other libraries, the benefits of switching may not outweigh the costs.