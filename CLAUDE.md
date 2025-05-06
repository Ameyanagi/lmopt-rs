# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Compiler Requirements

This project requires the Rust nightly compiler due to:

1. The use of advanced features in matrix computations and numeric algorithms
2. Potential future integration with developing autodiff features in Rust

A rust-toolchain.toml file is included in the repository to automatically select the nightly compiler.

### Automatic Differentiation Support

This project leverages Rust nightly's experimental `std::autodiff` module (powered by Enzyme) for automatic differentiation:

- **std::autodiff**: Nightly-only experimental API that performs automatic differentiation at the LLVM IR level

The `#[autodiff]` attribute macro has the syntax:
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

Example for a basic function f(x,y) = x² + 3y:
```rust
#[autodiff(df, Reverse, Active, Active, Active)]
fn f(x: f32, y: f32) -> f32 {
    x * x + 3.0 * y
}

// Generated df function returns (original_result, d_dx, d_dy)
// df(5.0, 7.0, 1.0) returns (46.0, 10.0, 3.0)
```

Our autodiff approach:
- Automatically computes derivatives for Jacobian matrices and gradients
- Intelligently selects forward or reverse mode based on problem dimensions
- Eliminates the need for manually coding derivatives
- Produces exact derivatives without numerical approximation errors
- Falls back to numerical differentiation when autodiff is not applicable

Our implementation strategy is tiered:
1. Use user-provided analytical derivatives when available (fastest)
2. Apply automatic differentiation when possible (accurate and reasonably fast)
3. Fall back to numerical differentiation when necessary (most general but least accurate)

For additional details, see the [autodiff.md](docs/autodiff.md) documentation file.

## Build/Test Commands

- Build: `cargo build`
- Run: `cargo run`
- Test all: `cargo test`
- Test specific: `cargo test test_name`
- Test specific module: `cargo test module_name`
- Lint: `cargo clippy -- -D warnings`
- Format: `cargo fmt`
- Benchmark: `cargo bench`
- Documentation: `cargo doc --open`

## Development Approach

- **Compatible Interface**: Implement a compatible interface with the levenberg-marquardt crate
- **Uncertainty**: Add uncertainty calculations similar to lmfit-py
- **Pure Rust**: Keep the implementation in pure Rust with no Python dependencies
- **Matrix Calculations**: Use faer for matrix operations internally, ndarray for public API
- **Matrix Interoperability**: Support both ndarray and faer matrices with efficient conversion
- **TDD**: Write failing tests first, then implement code to make them pass
- **Error Handling**: Use thiserror for library errors

## Implementation Plans

### Phase 1: Core Infrastructure (Current)
1. Set up project structure and dependencies
2. Implement matrix conversion utilities between ndarray, faer, and nalgebra
3. Create problem definition trait that maintains compatibility with levenberg-marquardt
4. Implement core LM algorithm with basic functionality

### Phase 2: Parameter System
1. Implement parameter system with bounds and constraints similar to lmfit-py
2. Add serialization/deserialization support for parameters
3. Develop parameter expressions and algebraic constraints

### Phase 3: Uncertainty Calculations
1. Implement confidence interval calculations
2. Add covariance matrix estimation
3. Develop Monte Carlo methods for uncertainty propagation
4. Create visualization utilities for confidence regions

### Phase 4: Model System
1. Implement base Model trait and abstract Model implementation
2. Create built-in models similar to lmfit-py (peak, step, etc.)
3. Add composite model support
4. Implement model serialization/deserialization

### Phase 5: Advanced Features
1. Add global optimization methods
2. Implement robust fitting methods
3. Create advanced constraint systems
4. Add parallel processing support for computationally intensive operations

## Implementation State Tracking

| Module | Status | Notes |
|--------|--------|-------|
| Matrix conversion | Not started | Planned for Phase 1 |
| Problem definition | Not started | Planned for Phase 1 |
| Core LM algorithm | Not started | Planned for Phase 1 |
| Parameter system | Not started | Planned for Phase 2 |
| Uncertainty calculations | Not started | Planned for Phase 3 |
| Model system | Not started | Planned for Phase 4 |
| Built-in models | Not started | Planned for Phase 4 |
| Global optimization | Not started | Planned for Phase 5 |

## AI-Assisted Development Process

This project uses AI-assisted development with structured planning and state management to ensure systematic progress and maintainable code.

### Plans Directory

The `plans/` directory contains detailed implementation plans for each phase of development:

- **implementation_plan.md**: Master plan for the entire project
- **phase1_plan.md through phase5_plan.md**: Detailed plans for each development phase
- Each plan includes:
  - Specific tasks with estimated timelines
  - Acceptance criteria for each task
  - Dependencies between tasks
  - Implementation approaches and algorithms
  - Testing strategies

### States Directory

The `.states/` directory tracks implementation progress using JSON files:

- **component_states.json**: Current status of each component (not started, in progress, completed, tested)
- **implementation_progress.json**: Overall progress tracking with completion percentages
- **test_coverage.json**: Test coverage metrics and quality assessments
- **backlog.json**: Features and tasks backlog with priorities

### Usage Guidelines

- Update state files after completing each task
- Review and revise plans based on implementation experience
- Use state tracking to report progress and identify bottlenecks
- Keep plans and states in sync with actual implementation

## Development Workflow

1. Study both levenberg-marquardt and lmfit-py to understand the interfaces and functionality
2. Create a detailed implementation plan before writing any code
3. Document the current state and intended outcome
4. For each component:
   - Design the API that is compatible with levenberg-marquardt while adding uncertainty features
   - Write failing tests that validate expected behavior against both reference implementations
   - Implement the code to make tests pass
   - Run tests to verify implementation
   - Format code: `cargo fmt`
   - Run linter: `cargo clippy -- -D warnings`
5. Before committing:
   - Run all tests
   - Fix any formatting, linting, or other issues
   - Carefully review the changes with `git diff`
   - Write a clear, descriptive commit message
6. Continuously update README.md with progress and usage instructions

## Documentation Specifications

### API Documentation
- All public APIs must have comprehensive rustdoc comments
- Mathematical functions must include:
  - The mathematical formula in standard notation
  - Citation of the algorithm source if applicable
  - Parameter descriptions with units
  - Return value description with units
  - Example usage

### User Guide Documentation
- Create a complete user guide in the project's documentation directory
- Include:
  - Getting started tutorials
  - Core concepts explanation
  - Example workflows
  - API reference
  - Performance considerations
  - Migration guides from levenberg-marquardt and lmfit-py

### Documentation Structure
- **docs/**
  - **getting_started.md** - Introduction and basic usage
  - **concepts/** - Core concept explanations
    - **lm_algorithm.md** - Explanation of Levenberg-Marquardt algorithm
    - **parameters.md** - Parameter system documentation
    - **uncertainty.md** - Uncertainty calculation methodology
    - **models.md** - Model system documentation
  - **examples/** - Documented examples
    - **basic_fitting.md** - Simple fitting examples
    - **parameter_constraints.md** - Using parameter constraints
    - **uncertainty_analysis.md** - Working with fit uncertainties
    - **custom_models.md** - Creating custom models
  - **api/** - Generated API documentation
  - **comparison/** - Comparison with reference implementations
    - **lm_crate.md** - Comparison with levenberg-marquardt crate
    - **lmfit_py.md** - Comparison with lmfit-py

## Reference Documentation

- Faer documentation: https://docs.rs/faer/latest/faer/
  - High-performance linear algebra library optimized for medium to large dense matrices
  - Provides matrix decompositions (Cholesky, LU, QR, SVD, Eigendecomposition)
  - Supports parallel processing via Rayon and SIMD acceleration
- levenberg-marquardt documentation: https://docs.rs/levenberg-marquardt/
  - Reference implementation we need to maintain compatibility with
- lmfit-py documentation: https://lmfit.github.io/lmfit-py/
  - Reference for uncertainty calculation features we want to implement

## Technical Specifications

### Levenberg-Marquardt Algorithm Implementation
- Trust region variant of the Levenberg-Marquardt algorithm
- Support for parameters with bounds and constraints
- Multiple options for linear algebra decompositions (QR, SVD, Cholesky)
- Step size control with adaptive damping parameter
- Configurable convergence criteria
- Robust fit statistics and diagnostics

### Parameter System Specifications
- Support for fixed and varying parameters
- Box constraints (min/max bounds)
- Algebraic constraints between parameters
- Parameter expressions with mathematical operations
- Derived parameters calculated from fitted parameters
- Serialization/deserialization support

### Uncertainty Calculation Specifications
- Covariance matrix estimation
- Confidence interval calculations
- Monte Carlo simulations for uncertainty propagation
- F-test for parameter significance
- Chi-square analysis
- Profile likelihood method for non-Gaussian uncertainties

### Model System Specifications
- Pluggable model interface
- Built-in standard models (Gaussian, Lorentzian, etc.)
- Composite model support (addition, multiplication, etc.)
- Custom model creation
- Model serialization/deserialization
- Automatic parameter initialization

## Library Dependencies

- faer = "0.22" - For high-performance linear algebra operations
- faer-ext = { version = "0.6", features = ["ndarray"] } - For conversion between faer and ndarray
- ndarray = "0.15" - For n-dimensional array operations (public API)
- thiserror = "1.0" - For library error type definitions
- rand = "0.8" - For random number generation in Monte Carlo methods
- serde = { version = "1.0", features = ["derive"] } - For serialization/deserialization
- serde_json = "1.0" - For JSON handling

## Matrix Types Usage

- Use ndarray for public API and interfaces to maintain user-friendly interactions
- Use faer matrices for all internal matrix operations and computations
- Leverage faer-ext for efficient conversion between ndarray and faer at API boundaries
- Provide conversion functions between nalgebra and faer/ndarray for levenberg-marquardt compatibility

## Directory Structure

- **src/**
  - **lib.rs** - Main library entry point and module declarations
  - **error.rs** - Library error definitions using thiserror
  - **problem.rs** - Problem definition trait and implementations
  - **lm.rs** - Main Levenberg-Marquardt algorithm implementation
  - **lm/**
    - **algorithm.rs** - Core algorithm steps
    - **trust_region.rs** - Trust region implementation
    - **convergence.rs** - Convergence criteria
    - **step.rs** - Step calculation and damping
  - **parameters.rs** - Parameter handling with bounds and constraints
  - **parameters/**
    - **bounds.rs** - Parameter bounds implementation
    - **constraints.rs** - Parameter constraints implementation
    - **expressions.rs** - Parameter expressions implementation
  - **uncertainty/** - Uncertainty calculation modules
    - **confidence.rs** - Confidence interval calculations
    - **covariance.rs** - Covariance matrix estimation
    - **monte_carlo.rs** - Monte Carlo uncertainty estimation
    - **profile.rs** - Profile likelihood method
  - **model.rs** - Base model trait and implementation
  - **models/** - Built-in model implementations similar to lmfit-py
    - **peak.rs** - Peak model functions (Gaussian, Lorentzian, etc.)
    - **step.rs** - Step model functions
    - **exponential.rs** - Exponential and power law models
    - **polynomial.rs** - Polynomial models
  - **utils/** - Common utilities and helper functions
    - **finite_difference.rs** - Numerical differentiation utilities
    - **matrix_convert.rs** - Conversion utilities between matrix types
    - **serialization.rs** - Serialization helpers
- **tests/**
  - **error_handling.rs** - Tests for error handling
  - **matrix_conversion.rs** - Tests for matrix conversion utilities
  - **problem_definition.rs** - Tests for problem definition implementation
  - **lm/** - Tests for Levenberg-Marquardt implementation
    - **algorithm.rs** - Tests for core algorithm
    - **trust_region.rs** - Tests for trust region implementation
    - **convergence.rs** - Tests for convergence criteria
  - **parameters/** - Tests for parameter system
    - **bounds.rs** - Tests for parameter bounds
    - **constraints.rs** - Tests for parameter constraints
    - **expressions.rs** - Tests for parameter expressions
  - **uncertainty/** - Tests for uncertainty calculations
    - **confidence.rs** - Tests for confidence intervals
    - **covariance.rs** - Tests for covariance matrix estimation
    - **monte_carlo.rs** - Tests for Monte Carlo methods
  - **models/** - Tests for model system
    - **peak.rs** - Tests for peak models
    - **step.rs** - Tests for step models
    - **composite.rs** - Tests for composite models
  - **integration/** - Integration tests
    - **nist_strd.rs** - NIST Statistical Reference Datasets
    - **lmfit_comparison.rs** - Comparison with lmfit-py results
- **examples/**
  - **basic_fitting.rs** - Simple fitting examples
  - **parameter_constraints.rs** - Using parameter constraints
  - **uncertainty_analysis.rs** - Working with fit uncertainties
  - **custom_models.rs** - Creating custom models
  - **composite_models.rs** - Working with composite models
  - **global_optimization.rs** - Global optimization examples
- **benches/**
  - **lm_algorithm.rs** - Benchmarks for LM algorithm
  - **matrix_operations.rs** - Benchmarks for matrix operations
  - **comparison.rs** - Benchmarks comparing to reference implementations
- **docs/** - Documentation files
  - See Documentation Structure section above
- **plans/** - Development planning and progression
  - **implementation_plan.md** - Detailed implementation plan
  - **phase1_plan.md** - Detailed plan for Phase 1
  - **phase2_plan.md** - Detailed plan for Phase 2
  - **phase3_plan.md** - Detailed plan for Phase 3
  - **phase4_plan.md** - Detailed plan for Phase 4
  - **phase5_plan.md** - Detailed plan for Phase 5
- **.states/** - State management for AI-assisted development
  - **component_states.json** - Current state of each component
  - **implementation_progress.json** - Overall implementation progress
  - **test_coverage.json** - Test coverage tracking
  - **backlog.json** - Features and tasks backlog

## Code Style Guidelines

- **Formatting**: Follow Rust standard formatting with `cargo fmt`
- **Naming**:
  - Use descriptive names for functions and variables
  - Follow Rust naming conventions (snake_case for functions and variables, CamelCase for types)
  - Include units in comments or documentation where relevant
  - For mathematical formulas, include comments with the standard equation notation
- **Organization**:
  - Split code into modules matching the original implementations while improving organization
  - Place internal utility functions in a dedicated `utils.rs` file
  - Define all error types in a dedicated `error.rs` file
- **Documentation**: All public APIs must have rustdoc comments with mathematical explanation
- **Error Types**: Define specific error types with thiserror
- **Performance**: Optimize numerically intensive operations, use faer efficiently
- **Testing**:
  - Unit test mathematical correctness against reference values from both reference implementations
  - Place small unit tests in each source file under `#[cfg(test)]` modules
  - Place larger integration tests in the `tests/` directory organized by module
- **Interfaces**: Create clean abstractions with idiomatic Rust APIs while maintaining compatibility

## Licensing

- This project is MIT-licensed
- Properly attribute and reference both levenberg-marquardt and lmfit-py
- Ensure all code follows the license requirements