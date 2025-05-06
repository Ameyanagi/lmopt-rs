# Test Plan for lmopt-rs

This document outlines the comprehensive testing strategy for the `lmopt-rs` project, focusing on ensuring the correctness, reliability, and performance of the Levenberg-Marquardt optimization implementation.

## Testing Goals

1. Verify correctness of matrix conversions between different linear algebra libraries
2. Ensure proper functionality of the Problem trait implementation
3. Validate the Levenberg-Marquardt algorithm's convergence properties
4. Compare results with reference implementations (levenberg-marquardt crate and lmfit-py)
5. Test edge cases and error handling
6. Measure performance and resource usage

## Testing Approach

### Unit Tests

Unit tests will focus on individual components to ensure they work correctly in isolation.

#### Matrix Conversion Tests

- Round-trip conversions between ndarray, faer, and nalgebra
- Handling of different matrix dimensions (square, rectangular, vectors)
- Edge cases: empty matrices, single-element matrices
- Performance for large matrices

#### Problem Trait Tests

- Correct definition and implementation of the Problem trait
- Expected behavior of derived methods (e.g., eval_cost)
- Testing with simple problems that have analytical solutions
- Edge cases: zero parameters, zero residuals
- Error handling for mismatched dimensions

#### Levenberg-Marquardt Algorithm Tests

- Convergence on standard test problems
- Behavior with different initial conditions
- Handling of ill-conditioned problems
- Testing different termination criteria
- Performance and scaling with problem size

### Integration Tests

Integration tests will validate the combined functionality of multiple components.

#### Standard Test Problems

- Linear regression: y = mx + b
- Nonlinear regression: exponential decay, Gaussian peak
- Rosenbrock function optimization
- NIST Statistical Reference Datasets (StRD)

#### Compatibility Tests

- Compatibility with levenberg-marquardt crate's Problem trait
- Comparison with lmfit-py results on identical problems

### Performance Tests

- Benchmark against reference implementations
- Resource usage (memory, CPU)
- Scaling properties with problem size

## Test Data

### Synthetic Test Data

- Simple functions with known analytical solutions
- Controlled noise levels for analyzing robustness
- Edge cases designed to stress specific aspects of the algorithm

### Real-world Test Data

- NIST StRD datasets for regression testing
- Application-specific datasets (spectroscopy, physics models, etc.)

## Test Categories

### Correctness Tests

Tests that verify the mathematical correctness of the implementation:

- Matrix operations correctness
- Gradient and Jacobian calculation accuracy
- Convergence to known solutions

### Robustness Tests

Tests that evaluate behavior in challenging conditions:

- Handling of poorly conditioned problems
- Sensitivity to numerical precision issues
- Recovery from suboptimal initial conditions

### Performance Tests

Tests that measure computational efficiency:

- Execution time for standard problems
- Memory usage patterns
- Scaling with problem complexity

## Implementation Plan

### Phase 1: Basic Unit Tests

1. Implement matrix conversion unit tests
   - Use small matrices with known values
   - Verify roundtrip conversion accuracy
   - Test edge cases

2. Implement Problem trait unit tests
   - Define simple test problems with analytical solutions
   - Test core functionality (eval, jacobian, etc.)
   - Test derived functionality (eval_cost)

3. Implement LM algorithm basic tests
   - Test convergence on simple problems
   - Verify step calculation and damping behavior
   - Test termination criteria

### Phase 2: Advanced Tests and Benchmarks

1. Implement more complex test problems
   - Multi-dimensional optimization
   - Problems with multiple local minima
   - Ill-conditioned problems

2. Develop comparison tests with reference implementations
   - Set up identical problems for levenberg-marquardt crate
   - Compare solutions and convergence behavior

3. Implement benchmarks
   - Measure performance against references
   - Profile critical code paths
   - Identify optimization opportunities

### Phase 3: Integration and Regression Tests

1. Create comprehensive integration tests
   - End-to-end workflows
   - Combined functionality tests

2. Implement regression tests
   - NIST reference datasets
   - Historical test cases

3. Set up continuous testing infrastructure
   - CI/CD integration
   - Automated regression testing

## Test Case Templates

### Matrix Conversion Test Template

```rust
#[test]
fn test_matrix_conversion_roundtrip() {
    // Create test matrix with known values
    let original = /* define matrix */;
    
    // Convert to target format and back
    let converted = /* first conversion */;
    let roundtrip = /* second conversion */;
    
    // Verify results
    assert_eq!(original, roundtrip);
}
```

### Problem Implementation Test Template

```rust
#[test]
fn test_problem_evaluation() {
    // Define test problem with known solution
    let problem = TestProblem::new();
    
    // Evaluate at test points
    let params = /* test parameters */;
    let residuals = problem.eval(&params).unwrap();
    
    // Verify results match expected values
    assert_relative_eq!(residuals, expected_residuals);
}
```

### LM Algorithm Test Template

```rust
#[test]
fn test_lm_convergence() {
    // Define test problem
    let problem = TestProblem::new();
    
    // Set initial parameters
    let initial_params = /* initial guess */;
    
    // Run optimization
    let lm = LevenbergMarquardt::with_default_config();
    let result = lm.minimize(&problem, initial_params).unwrap();
    
    // Verify convergence to expected solution
    assert!(result.success);
    assert_relative_eq!(result.params, expected_solution, epsilon = 1e-6);
}
```

## Special Test Considerations

### Numerical Stability

- Test with varying precision (f32/f64)
- Test with ill-conditioned matrices
- Verify behavior near singularities

### Edge Cases

- Zero or negative damping parameters
- Parameters at or near bounds
- Problems with no unique solution
- Extremely flat or steep objective functions

### Performance Considerations

- Test with both small and large problems
- Measure memory usage patterns
- Evaluate parallel execution benefits

## Key Test Cases

1. **Linear Regression Test**: Fit a straight line to noisy data
2. **Exponential Decay Test**: Fit an exponential decay function
3. **Gaussian Peak Test**: Fit a Gaussian peak model to simulated data
4. **Polynomial Fit Test**: Fit polynomials of various degrees
5. **Rosenbrock Function Test**: Minimize the notoriously difficult Rosenbrock function
6. **Bennett5 NIST Problem**: Standard reference problem with known solution
7. **Hahn1 NIST Problem**: Another reference problem with known solution

## Success Criteria

- All unit tests pass
- Integration tests match reference implementations within acceptable tolerance
- Performance benchmarks meet or exceed reference implementations
- Edge cases are handled gracefully with appropriate error messages
- Code coverage exceeds 90% for core components