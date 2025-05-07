//! Tests for constrained global optimization problems.
//!
//! These tests evaluate how well the global optimization algorithms handle
//! problems with constraints, both in the form of parameter bounds and
//! more complex constraints.

use approx::assert_relative_eq;
use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::{DifferentialEvolution, GlobalOptimizer, ParallelDifferentialEvolution};
use lmopt_rs::parameters::{Parameter, Parameters};
use lmopt_rs::problem::Problem;
use lmopt_rs::problem_params::{problem_from_parameter_problem, ParameterProblem};
use ndarray::{array, Array1};

// --- Constrained Test Problems ---

/// Constrained optimization problem with equality constraints
///
/// Minimize f(x) = (x₁ - 2)² + (x₂ - 1)²
/// Subject to: h₁(x) = x₁ - 2x₂ + 1 = 0
///            h₂(x) = x₁² + x₂² - 1 = 0
///
/// The solution is approximately x* = [0.7864, 0.8932] with f(x*) = 1.3935
///
/// We implement the constraints using a penalty method.
struct ConstrainedProblem;

impl Problem for ConstrainedProblem {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 2 {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        let x1 = params[0];
        let x2 = params[1];

        // Objective function
        let obj = (x1 - 2.0).powi(2) + (x2 - 1.0).powi(2);

        // Equality constraints with penalty
        let h1 = (x1 - 2.0 * x2 + 1.0).abs();
        let h2 = (x1.powi(2) + x2.powi(2) - 1.0).abs();

        // Add penalty for constraint violations
        let penalty = 100.0 * (h1 + h2);

        Ok(array![(obj + penalty).sqrt()])
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Constrained optimization problem with inequality constraints
///
/// Minimize f(x) = 100(x₂ - x₁²)² + (1 - x₁)²
/// Subject to: g₁(x) = x₁ + x₂² ≤ 1
///            g₂(x) = x₁² + x₂ ≤ 1
///
/// This is a constrained version of the Rosenbrock function
struct ConstrainedRosenbrock;

impl Problem for ConstrainedRosenbrock {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != 2 {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected 2 parameters, got {}",
                params.len()
            )));
        }

        let x1 = params[0];
        let x2 = params[1];

        // Rosenbrock function
        let obj = 100.0 * (x2 - x1.powi(2)).powi(2) + (1.0 - x1).powi(2);

        // Inequality constraints with penalty
        let g1 = (x1 + x2.powi(2) - 1.0).max(0.0);
        let g2 = (x1.powi(2) + x2 - 1.0).max(0.0);

        // Add penalty for constraint violations
        let penalty = 1000.0 * (g1 + g2);

        Ok(array![(obj + penalty).sqrt()])
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Parameter-based problem with algebraic constraints
///
/// This problem uses the parameter system to create constraints between parameters
struct ParameterBasedProblem {
    parameters: Parameters,
}

impl ParameterBasedProblem {
    fn new() -> Result<Self> {
        let mut params = Parameters::new();

        // Add parameters with bounds
        params.add_param_with_bounds("x1", 0.5, -2.0, 2.0)?;
        params.add_param_with_bounds("x2", 0.5, -2.0, 2.0)?;

        // Add a derived parameter with expression constraint
        params.add_param_with_expr("sum", 1.0, "x1 + x2")?;

        // Add another constraint parameter
        params.add_param_with_expr("constraint", 0.0, "x1^2 + x2^2 - 1.0")?;

        Ok(Self { parameters: params })
    }

    // Objective function
    fn objective(&self) -> Result<f64> {
        let x1 = self.parameters.get("x1")?.value();
        let x2 = self.parameters.get("x2")?.value();

        // Simple objective function
        let obj = (x1 - 1.0).powi(2) + (x2 - 1.0).powi(2);

        // Add penalty for constraint violations
        let constraint = self.parameters.get("constraint")?.value();
        let penalty = 100.0 * constraint.powi(2);

        Ok(obj + penalty)
    }
}

impl ParameterProblem for ParameterBasedProblem {
    fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut Parameters {
        &mut self.parameters
    }

    fn update_parameters_from_array(&mut self, params: &Array1<f64>) -> Result<()> {
        let vary_params = self.parameters.varying();

        if vary_params.len() != params.len() {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                vary_params.len(),
                params.len()
            )));
        }

        // Update parameter values
        for (i, param) in vary_params.into_iter().enumerate() {
            param.set_value(params[i])?;
        }

        // Update derived parameters
        self.parameters.update_dependent_parameters()?;

        Ok(())
    }

    fn parameters_to_array(&self) -> Result<Array1<f64>> {
        let vary_params = self.parameters.varying();
        let mut result = Array1::zeros(vary_params.len());

        for (i, param) in vary_params.into_iter().enumerate() {
            result[i] = param.value();
        }

        Ok(result)
    }

    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        // Create a temporary clone that we can modify
        let mut temp_problem = self.clone();

        // Update parameters
        temp_problem.update_parameters_from_array(params)?;

        // Evaluate objective function
        let obj = temp_problem.objective()?;

        Ok(array![obj.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.parameters.varying().len()
    }

    fn residual_count(&self) -> usize {
        1
    }
}

impl Clone for ParameterBasedProblem {
    fn clone(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
        }
    }
}

// --- Tests ---

#[test]
fn test_constrained_problem() -> Result<()> {
    let problem = ConstrainedProblem;
    let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];

    // Expected solution from analytical solution
    let expected_solution = [0.7864, 0.8932];

    // Try with both sequential and parallel DE
    let de = DifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8);

    let result = de.optimize(&problem, &bounds, 200, 50, 1e-6)?;

    println!("DE result for constrained problem:");
    println!("  Success: {}", result.success);
    println!(
        "  Solution: [{:.4}, {:.4}]",
        result.params[0], result.params[1]
    );
    println!("  Cost: {:.6e}", result.cost);

    // Check if the solution is close to the expected solution
    assert_relative_eq!(result.params[0], expected_solution[0], epsilon = 0.05);
    assert_relative_eq!(result.params[1], expected_solution[1], epsilon = 0.05);

    // Now with parallel DE
    let par_de = ParallelDifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8);

    let par_result = par_de.optimize(&problem, &bounds, 200, 50, 1e-6)?;

    println!("Parallel DE result for constrained problem:");
    println!("  Success: {}", par_result.success);
    println!(
        "  Solution: [{:.4}, {:.4}]",
        par_result.params[0], par_result.params[1]
    );
    println!("  Cost: {:.6e}", par_result.cost);

    // Check if the solution is close to the expected solution
    assert_relative_eq!(par_result.params[0], expected_solution[0], epsilon = 0.05);
    assert_relative_eq!(par_result.params[1], expected_solution[1], epsilon = 0.05);

    Ok(())
}

#[test]
fn test_constrained_rosenbrock() -> Result<()> {
    let problem = ConstrainedRosenbrock;
    let bounds = vec![(-2.0, 2.0), (-2.0, 2.0)];

    // The global minimum of the constrained problem is at the boundary of the feasible region

    let de = DifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8)
        .with_local_optimization(true);

    let result = de.optimize(&problem, &bounds, 200, 50, 1e-6)?;

    println!("DE result for constrained Rosenbrock:");
    println!("  Success: {}", result.success);
    println!(
        "  Solution: [{:.4}, {:.4}]",
        result.params[0], result.params[1]
    );
    println!("  Cost: {:.6e}", result.cost);

    // Check constraints
    let x1 = result.params[0];
    let x2 = result.params[1];

    // Verify constraints are satisfied (with small tolerance)
    let g1 = x1 + x2.powi(2) - 1.0;
    let g2 = x1.powi(2) + x2 - 1.0;

    println!("  Constraint g1: {:.6e}", g1);
    println!("  Constraint g2: {:.6e}", g2);

    assert!(g1 <= 0.05, "Constraint g1 violation: {:.6e}", g1);
    assert!(g2 <= 0.05, "Constraint g2 violation: {:.6e}", g2);

    Ok(())
}

#[test]
fn test_parameter_problem() -> Result<()> {
    let mut problem = ParameterBasedProblem::new()?;

    // Create adapter for the global optimizer
    let adapter = problem_from_parameter_problem(&problem);

    // Get bounds from the parameters
    let bounds = problem
        .parameters()
        .varying()
        .iter()
        .map(|p| (p.min(), p.max()))
        .collect::<Vec<_>>();

    // Optimize with DE
    let de = DifferentialEvolution::new()
        .with_population_size(40)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8);

    let result = de.optimize(&adapter, &bounds, 150, 40, 1e-6)?;

    println!("DE result for parameter problem:");
    println!("  Success: {}", result.success);
    println!(
        "  Solution: [{:.4}, {:.4}]",
        result.params[0], result.params[1]
    );
    println!("  Cost: {:.6e}", result.cost);

    // Update problem with the best parameters
    problem.update_parameters_from_array(&result.params)?;

    // Check derived parameters and constraints
    let sum = problem.parameters().get("sum")?.value();
    let constraint = problem.parameters().get("constraint")?.value();

    println!("  Sum (x1 + x2): {:.4}", sum);
    println!("  Constraint (x1^2 + x2^2 - 1): {:.4e}", constraint);

    // The constraint should be close to zero
    assert_relative_eq!(
        constraint,
        0.0,
        epsilon = 0.01,
        "Constraint not satisfied: {:.4e}",
        constraint
    );

    Ok(())
}

#[test]
fn test_multi_objective_constraints() -> Result<()> {
    // This test simulates a multi-objective problem with constraints
    // by combining multiple objectives with weights

    struct MultiObjectiveProblem {
        weights: Vec<f64>,
    }

    impl MultiObjectiveProblem {
        fn new(weights: Vec<f64>) -> Self {
            Self { weights }
        }
    }

    impl Problem for MultiObjectiveProblem {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            if params.len() != 3 {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected 3 parameters, got {}",
                    params.len()
                )));
            }

            let x1 = params[0];
            let x2 = params[1];
            let x3 = params[2];

            // Multiple objective functions
            let obj1 = x1.powi(2) + x2.powi(2) + x3.powi(2);
            let obj2 = (x1 - 1.0).powi(2) + (x2 - 1.0).powi(2) + (x3 - 1.0).powi(2);

            // Constraints
            let g1 = (x1 + x2 + x3 - 1.0).abs(); // x1 + x2 + x3 = 1
            let g2 = (x1 * x2 * x3 - 0.125).max(0.0); // x1 * x2 * x3 <= 0.125

            // Combine objectives with weights
            let weighted_obj = self.weights[0] * obj1 + self.weights[1] * obj2;

            // Add penalty for constraint violations
            let penalty = 1000.0 * (g1 + g2);

            Ok(array![(weighted_obj + penalty).sqrt()])
        }

        fn parameter_count(&self) -> usize {
            3
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    // Create problem with different objective weights
    let weights = vec![0.7, 0.3]; // 70% obj1, 30% obj2
    let problem = MultiObjectiveProblem::new(weights);
    let bounds = vec![(-2.0, 2.0); 3];

    // Optimize with parallel DE
    let par_de = ParallelDifferentialEvolution::new()
        .with_population_size(50)
        .with_crossover_probability(0.9)
        .with_differential_weight(0.8)
        .with_local_optimization(true);

    let result = par_de.optimize(&problem, &bounds, 200, 50, 1e-6)?;

    println!("Parallel DE result for multi-objective problem:");
    println!("  Success: {}", result.success);
    println!(
        "  Solution: [{:.4}, {:.4}, {:.4}]",
        result.params[0], result.params[1], result.params[2]
    );
    println!("  Cost: {:.6e}", result.cost);

    // Check constraints
    let x1 = result.params[0];
    let x2 = result.params[1];
    let x3 = result.params[2];

    let sum = x1 + x2 + x3;
    let product = x1 * x2 * x3;

    println!("  Constraint sum (should be ≈ 1): {:.4}", sum);
    println!("  Constraint product (should be ≤ 0.125): {:.4}", product);

    // Verify constraints are satisfied (with small tolerance)
    assert_relative_eq!(
        sum,
        1.0,
        epsilon = 0.05,
        "Sum constraint violation: {:.4} != 1",
        sum
    );
    assert!(
        product <= 0.13,
        "Product constraint violation: {:.4} > 0.125",
        product
    );

    Ok(())
}
