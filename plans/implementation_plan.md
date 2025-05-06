# Implementation Plan for lmopt-rs

This document outlines the detailed implementation plan for the `lmopt-rs` project, which aims to create a Rust library for nonlinear least-squares optimization using the Levenberg-Marquardt algorithm with uncertainty calculation capabilities.

## Project Goals

1. Create a Rust implementation of the Levenberg-Marquardt algorithm that is compatible with the existing `levenberg-marquardt` crate
2. Add uncertainty calculation capabilities similar to those found in `lmfit-py`
3. Provide a clean, idiomatic Rust API with comprehensive documentation
4. Ensure high performance using the `faer` matrix library internally
5. Support both `ndarray` and `faer` matrix types with efficient conversion

## Implementation Phases

### Phase 1: Core Infrastructure

#### 1.1 Project Structure and Dependencies

- [x] Set up basic project structure
- [x] Configure Cargo.toml with dependencies
- [ ] Create initial module structure
- [ ] Add error handling with thiserror

#### 1.2 Matrix Conversion Utilities

- [ ] Implement conversion between ndarray and faer
- [ ] Implement conversion between nalgebra and faer
- [ ] Implement conversion between nalgebra and ndarray
- [ ] Add unit tests for matrix conversions

#### 1.3 Problem Definition

- [ ] Define Problem trait compatible with levenberg-marquardt crate
- [ ] Implement adapter for levenberg-marquardt Problem implementations
- [ ] Add unit tests for Problem trait implementations

#### 1.4 Core LM Algorithm

- [ ] Implement basic LM algorithm steps using faer
- [ ] Add trust region implementation
- [ ] Implement convergence criteria
- [ ] Add step calculation with damping parameter
- [ ] Implement model evaluation and residual calculation
- [ ] Add unit tests for LM algorithm components
- [ ] Create integration tests comparing results with levenberg-marquardt

### Phase 2: Parameter System

#### 2.1 Basic Parameter Implementation

- [ ] Create Parameter struct with name, value, vary flag
- [ ] Add bounds constraints (min, max)
- [ ] Implement parameter expressions
- [ ] Add serialization/deserialization

#### 2.2 Advanced Parameter Features

- [ ] Implement algebraic constraints between parameters
- [ ] Add derived parameters
- [ ] Create parameter handling utilities

#### 2.3 Testing

- [ ] Add unit tests for parameter functionality
- [ ] Create integration tests comparing with lmfit-py parameter handling

### Phase 3: Uncertainty Calculations

#### 3.1 Covariance Matrix Estimation

- [ ] Implement covariance matrix calculation
- [ ] Add confidence interval estimation
- [ ] Implement standard error calculation
- [ ] Add unit tests for covariance estimation

#### 3.2 Advanced Uncertainty Methods

- [ ] Implement Monte Carlo methods for uncertainty propagation
- [ ] Add profile likelihood method
- [ ] Implement F-test for parameter significance
- [ ] Create visualization utilities for confidence regions

#### 3.3 Testing

- [ ] Add unit tests for uncertainty calculations
- [ ] Create integration tests comparing with lmfit-py uncertainty estimates

### Phase 4: Model System

#### 4.1 Base Model Implementation

- [ ] Create Model trait and abstract implementation
- [ ] Add model evaluation and residual calculation
- [ ] Implement gradient calculation (analytical and numerical)
- [ ] Add serialization/deserialization

#### 4.2 Built-in Models

- [ ] Implement peak models (Gaussian, Lorentzian, etc.)
- [ ] Add step models
- [ ] Implement exponential and power law models
- [ ] Add polynomial models

#### 4.3 Composite Models

- [ ] Implement model composition (addition, multiplication)
- [ ] Add parameter sharing between models
- [ ] Create utilities for model selection and comparison

#### 4.4 Testing

- [ ] Add unit tests for model implementations
- [ ] Create integration tests comparing with lmfit-py models

### Phase 5: Advanced Features

#### 5.1 Global Optimization

- [ ] Implement differential evolution
- [ ] Add simulated annealing
- [ ] Implement basin-hopping

#### 5.2 Robust Fitting Methods

- [ ] Add robust loss functions
- [ ] Implement weighted fitting

#### 5.3 Performance Optimization

- [ ] Add parallel processing support
- [ ] Optimize critical code paths
- [ ] Implement caching for repeated evaluations

#### 5.4 Documentation and Examples

- [ ] Complete API documentation
- [ ] Create user guide
- [ ] Add comprehensive examples
- [ ] Create benchmarks and performance comparisons

## Timeline

| Phase | Duration | Target Completion |
|-------|----------|-------------------|
| Phase 1: Core Infrastructure | 4 weeks | Week 4 |
| Phase 2: Parameter System | 3 weeks | Week 7 |
| Phase 3: Uncertainty Calculations | 3 weeks | Week 10 |
| Phase 4: Model System | 3 weeks | Week 13 |
| Phase 5: Advanced Features | 3 weeks | Week 16 |
| Buffer and Final Integration | 2 weeks | Week 18 |

## Acceptance Criteria

### Phase 1 Acceptance

- Core LM algorithm passes all tests against reference data
- Results match levenberg-marquardt crate for the same problems
- Matrix conversions are efficient and accurate
- Problem trait is compatible with levenberg-marquardt

### Phase 2 Acceptance

- Parameter system handles bounds and constraints correctly
- Parameter expressions are evaluated correctly
- Serialization/deserialization works as expected

### Phase 3 Acceptance

- Uncertainty calculations match reference implementation (lmfit-py)
- Confidence intervals are calculated correctly
- Monte Carlo methods produce valid uncertainty estimates

### Phase 4 Acceptance

- Model system is flexible and extensible
- Built-in models match reference implementations
- Composite models work correctly

### Phase 5 Acceptance

- Global optimization methods converge to correct solutions
- Robust fitting methods handle outliers correctly
- Performance meets or exceeds reference implementations

## Implementation Strategy

- Test-driven development: Write tests first, then implement to pass tests
- Reference-driven: Compare results with levenberg-marquardt and lmfit-py
- Incremental implementation: Build and test in small, manageable pieces
- Regular benchmarking: Ensure performance targets are met
- Comprehensive documentation: Document as we go, with examples

## Dependencies

- Rust nightly toolchain
- faer and ndarray for linear algebra
- thiserror for error handling
- serde for serialization/deserialization
- rand for Monte Carlo methods
- std::autodiff for automatic differentiation (nightly feature)

Note: Rust nightly is required for:
1. The std::autodiff feature used for automatic differentiation
2. Advanced features in matrix computations
3. Various performance optimizations