//! Convergence criteria for optimization algorithms.
//!
//! This module defines the criteria used to determine when an optimization
//! algorithm has converged to a solution.

use ndarray::Array1;

/// Possible convergence states for an optimization algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// The algorithm is still running.
    Running,

    /// The algorithm has converged due to a small parameter change.
    ParameterConvergence,

    /// The algorithm has converged due to a small function value change.
    FunctionValueConvergence,

    /// The algorithm has converged due to a small gradient.
    GradientConvergence,

    /// The algorithm has terminated due to reaching the maximum number of iterations.
    MaxIterationsReached,

    /// The algorithm has terminated due to a numerical error.
    NumericalError,
}

impl ConvergenceStatus {
    /// Returns true if the optimization has terminated (either converged or failed).
    pub fn is_terminated(&self) -> bool {
        match self {
            ConvergenceStatus::Running => false,
            _ => true,
        }
    }

    /// Returns true if the optimization has converged.
    pub fn is_converged(&self) -> bool {
        match self {
            ConvergenceStatus::ParameterConvergence => true,
            ConvergenceStatus::FunctionValueConvergence => true,
            ConvergenceStatus::GradientConvergence => true,
            _ => false,
        }
    }

    /// Returns a description of the convergence status.
    pub fn description(&self) -> String {
        match self {
            ConvergenceStatus::Running => "Optimization is still running".to_string(),
            ConvergenceStatus::ParameterConvergence => {
                "Converged: small parameter change".to_string()
            }
            ConvergenceStatus::FunctionValueConvergence => {
                "Converged: small function value change".to_string()
            }
            ConvergenceStatus::GradientConvergence => "Converged: small gradient".to_string(),
            ConvergenceStatus::MaxIterationsReached => {
                "Terminated: maximum iterations reached".to_string()
            }
            ConvergenceStatus::NumericalError => "Terminated: numerical error".to_string(),
        }
    }
}

/// Criteria for determining when an optimization algorithm has converged.
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Tolerance for change in parameter values.
    pub xtol: f64,

    /// Tolerance for change in function value.
    pub ftol: f64,

    /// Tolerance for gradient norm.
    pub gtol: f64,

    /// Maximum number of iterations.
    pub max_iterations: usize,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            xtol: 1e-8,
            ftol: 1e-8,
            gtol: 1e-8,
            max_iterations: 100,
        }
    }
}

impl ConvergenceCriteria {
    /// Creates a new set of convergence criteria with the given tolerances.
    pub fn new(xtol: f64, ftol: f64, gtol: f64, max_iterations: usize) -> Self {
        Self {
            xtol,
            ftol,
            gtol,
            max_iterations,
        }
    }

    /// Checks whether the optimization has converged based on the current state.
    ///
    /// # Arguments
    ///
    /// * `params` - The current parameter values
    /// * `new_params` - The new parameter values
    /// * `cost` - The current function value
    /// * `new_cost` - The new function value
    /// * `gradient_norm` - The norm of the gradient
    /// * `iterations` - The number of iterations so far
    ///
    /// # Returns
    ///
    /// * The convergence status
    pub fn check(
        &self,
        params: &Array1<f64>,
        new_params: &Array1<f64>,
        cost: f64,
        new_cost: f64,
        gradient_norm: f64,
        iterations: usize,
    ) -> ConvergenceStatus {
        // Check iterations
        if iterations >= self.max_iterations {
            return ConvergenceStatus::MaxIterationsReached;
        }

        // Check gradient
        if gradient_norm < self.gtol {
            return ConvergenceStatus::GradientConvergence;
        }

        // Check parameter change
        let param_change_vec = new_params
            .iter()
            .zip(params.iter())
            .map(|(a, b)| (a - b).abs() / (b.abs().max(1.0)))
            .collect::<Vec<f64>>();
        let param_change = param_change_vec.iter().fold(0.0, |a, &b| a.max(b));
        if param_change < self.xtol {
            return ConvergenceStatus::ParameterConvergence;
        }

        // Check function value change
        let cost_change = (cost - new_cost).abs() / cost.max(1e-10);
        if cost_change < self.ftol {
            return ConvergenceStatus::FunctionValueConvergence;
        }

        ConvergenceStatus::Running
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_convergence_criteria() {
        let criteria = ConvergenceCriteria::default();

        // Test parameter convergence
        let params = array![1.0, 2.0, 3.0];
        let new_params = array![1.0000001, 2.0000001, 3.0000001];
        let status = criteria.check(&params, &new_params, 10.0, 9.9, 0.1, 50);
        assert_eq!(status, ConvergenceStatus::ParameterConvergence);

        // Test function value convergence
        let params = array![1.0, 2.0, 3.0];
        let new_params = array![1.1, 2.1, 3.1];
        let status = criteria.check(&params, &new_params, 10.0, 9.9999999, 0.1, 50);
        assert_eq!(status, ConvergenceStatus::FunctionValueConvergence);

        // Test gradient convergence
        let params = array![1.0, 2.0, 3.0];
        let new_params = array![1.1, 2.1, 3.1];
        let status = criteria.check(&params, &new_params, 10.0, 9.0, 1e-9, 50);
        assert_eq!(status, ConvergenceStatus::GradientConvergence);

        // Test max iterations
        let params = array![1.0, 2.0, 3.0];
        let new_params = array![1.1, 2.1, 3.1];
        let status = criteria.check(&params, &new_params, 10.0, 9.0, 0.1, 100);
        assert_eq!(status, ConvergenceStatus::MaxIterationsReached);

        // Test still running
        let params = array![1.0, 2.0, 3.0];
        let new_params = array![1.1, 2.1, 3.1];
        let status = criteria.check(&params, &new_params, 10.0, 9.0, 0.1, 50);
        assert_eq!(status, ConvergenceStatus::Running);
    }

    #[test]
    fn test_convergence_status_methods() {
        assert!(!ConvergenceStatus::Running.is_terminated());
        assert!(ConvergenceStatus::ParameterConvergence.is_terminated());
        assert!(ConvergenceStatus::FunctionValueConvergence.is_terminated());
        assert!(ConvergenceStatus::GradientConvergence.is_terminated());
        assert!(ConvergenceStatus::MaxIterationsReached.is_terminated());
        assert!(ConvergenceStatus::NumericalError.is_terminated());

        assert!(!ConvergenceStatus::Running.is_converged());
        assert!(ConvergenceStatus::ParameterConvergence.is_converged());
        assert!(ConvergenceStatus::FunctionValueConvergence.is_converged());
        assert!(ConvergenceStatus::GradientConvergence.is_converged());
        assert!(!ConvergenceStatus::MaxIterationsReached.is_converged());
        assert!(!ConvergenceStatus::NumericalError.is_converged());
    }
}
