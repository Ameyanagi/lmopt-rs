//! Implementation of algebraic constraints for parameters
//!
//! This module provides the ability to add algebraic constraints between parameters,
//! similar to lmfit-py's constraint system. Constraints are implemented using expressions
//! and can include equalities, inequalities, and more complex relationships.

use crate::parameters::expression::{Expression, ExpressionError};
use crate::parameters::parameter::ParameterError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error that can occur when working with constraints
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConstraintError {
    #[error("Invalid constraint expression: {message}")]
    InvalidExpression { message: String },

    #[error("Failed to evaluate constraint: {message}")]
    EvaluationError { message: String },

    #[error("Constraint violation: {message}")]
    ConstraintViolation { message: String },

    #[error("Parameter error: {0}")]
    ParameterError(#[from] ParameterError),

    #[error("Expression error: {0}")]
    ExpressionError(#[from] ExpressionError),
}

/// Type of constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Equal to (==)
    Equal,

    /// Not equal to (!=)
    NotEqual,

    /// Less than (<)
    LessThan,

    /// Less than or equal to (<=)
    LessThanOrEqual,

    /// Greater than (>)
    GreaterThan,

    /// Greater than or equal to (>=)
    GreaterThanOrEqual,
}

impl ConstraintType {
    /// Convert the constraint type to a string operator
    pub fn as_operator(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
        }
    }

    /// Check if the constraint is satisfied
    pub fn is_satisfied(&self, value: f64) -> bool {
        match self {
            Self::Equal => value.abs() < 1e-10, // lhs ≈ rhs (approx equal)
            Self::NotEqual => value.abs() >= 1e-10, // lhs ≠ rhs
            Self::LessThan => value < 0.0,      // lhs < rhs -> (lhs - rhs) < 0
            Self::LessThanOrEqual => value <= 0.0, // lhs ≤ rhs -> (lhs - rhs) <= 0
            Self::GreaterThan => value < 0.0,   // lhs > rhs -> (rhs - lhs) < 0
            Self::GreaterThanOrEqual => value <= 0.0, // lhs ≥ rhs -> (rhs - lhs) <= 0
        }
    }
}

/// A constraint between parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The left-hand side of the constraint (expression)
    pub lhs: String,

    /// The type of constraint (==, !=, <, <=, >, >=)
    pub constraint_type: ConstraintType,

    /// The right-hand side of the constraint (expression)
    pub rhs: String,

    /// The combined expression representing the constraint
    ///
    /// For example, "x < y" would be represented as "y - x"
    /// The constraint is satisfied if the condition matches the constraint type:
    /// - Equal: expr == 0
    /// - NotEqual: expr != 0
    /// - LessThan: expr < 0
    /// - LessThanOrEqual: expr <= 0
    /// - GreaterThan: expr > 0
    /// - GreaterThanOrEqual: expr >= 0
    #[serde(skip)]
    expr: Option<Expression>,
}

impl Constraint {
    /// Create a new constraint between two expressions
    pub fn new(
        lhs: &str,
        constraint_type: ConstraintType,
        rhs: &str,
    ) -> Result<Self, ConstraintError> {
        // Combine the expressions based on the constraint type
        // For all types, we create an expression where the constraint is satisfied if:
        // - Equal: expr ≈ 0 (very close to zero)
        // - NotEqual: expr ≠ 0 (not close to zero)
        // - LessThan: expr < 0
        // - LessThanOrEqual: expr ≤ 0
        // - GreaterThan: expr > 0
        // - GreaterThanOrEqual: expr ≥ 0
        let combined_expr = match constraint_type {
            // For Equal and NotEqual, we want lhs - rhs to be 0 or not 0
            ConstraintType::Equal => format!("({lhs}) - ({rhs})"),
            ConstraintType::NotEqual => format!("({lhs}) - ({rhs})"),

            // For LessThan and LessThanOrEqual, we want lhs - rhs < 0 or lhs - rhs ≤ 0
            ConstraintType::LessThan => format!("({lhs}) - ({rhs})"),
            ConstraintType::LessThanOrEqual => format!("({lhs}) - ({rhs})"),

            // For GreaterThan and GreaterThanOrEqual, we need to swap the order to rhs - lhs
            // This makes constraint satisfaction logic work correctly:
            // - lhs > rhs  is satisfied when (rhs - lhs) < 0
            // - lhs >= rhs is satisfied when (rhs - lhs) <= 0
            ConstraintType::GreaterThan => format!("({rhs}) - ({lhs})"),
            ConstraintType::GreaterThanOrEqual => format!("({rhs}) - ({lhs})"),
        };

        // Parse the combined expression
        let expr =
            Expression::parse(&combined_expr).map_err(|e| ConstraintError::InvalidExpression {
                message: format!("Failed to parse constraint: {}", e),
            })?;

        Ok(Self {
            lhs: lhs.to_string(),
            constraint_type,
            rhs: rhs.to_string(),
            expr: Some(expr),
        })
    }

    /// Get the combined expression for this constraint
    pub fn expression(&self) -> Option<&Expression> {
        self.expr.as_ref()
    }

    /// Get the variables used in this constraint
    pub fn variables(&self) -> Vec<String> {
        match &self.expr {
            Some(expr) => expr.variables(),
            None => {
                // If the expression hasn't been parsed yet, try to parse it
                if let Ok(expr) = self.get_or_parse_expr() {
                    expr.variables()
                } else {
                    Vec::new()
                }
            }
        }
    }

    /// Get the parsed expression or parse it if it hasn't been parsed yet
    pub fn get_or_parse_expr(&self) -> Result<&Expression, ConstraintError> {
        // If we already have a parsed expression, return it
        if let Some(expr) = &self.expr {
            return Ok(expr);
        }

        // Otherwise, this is an error - the expression should have been parsed in new()
        Err(ConstraintError::InvalidExpression {
            message: "Constraint expression not initialized".to_string(),
        })
    }

    /// Check if the constraint is satisfied
    pub fn is_satisfied<C>(&self, context: &C) -> Result<bool, ConstraintError>
    where
        C: crate::parameters::expression::EvaluationContext,
    {
        // Get the expression
        let expr = self.get_or_parse_expr()?;

        // Evaluate the expression
        let value = expr
            .evaluate(context)
            .map_err(|e| ConstraintError::EvaluationError {
                message: format!("Failed to evaluate constraint: {}", e),
            })?;

        // For debugging
        // println!("Constraint: {} {} {}, Expression: {:?}, Value: {}",
        //          self.lhs, self.constraint_type.as_operator(), self.rhs,
        //          expr, value);

        // Check if the constraint is satisfied based on its type
        Ok(self.constraint_type.is_satisfied(value))
    }

    /// Format the constraint as a string
    pub fn to_string(&self) -> String {
        format!(
            "{} {} {}",
            self.lhs,
            self.constraint_type.as_operator(),
            self.rhs
        )
    }
}

/// A collection of constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constraints {
    /// The list of constraints
    constraints: Vec<Constraint>,
}

impl Constraints {
    /// Create a new empty collection of constraints
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to the collection
    pub fn add(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add a constraint from expressions and constraint type
    pub fn add_constraint(
        &mut self,
        lhs: &str,
        constraint_type: ConstraintType,
        rhs: &str,
    ) -> Result<(), ConstraintError> {
        let constraint = Constraint::new(lhs, constraint_type, rhs)?;
        self.add(constraint);
        Ok(())
    }

    /// Remove a constraint from the collection
    pub fn remove(&mut self, index: usize) -> Option<Constraint> {
        if index < self.constraints.len() {
            Some(self.constraints.remove(index))
        } else {
            None
        }
    }

    /// Get the number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get a constraint by index
    pub fn get(&self, index: usize) -> Option<&Constraint> {
        self.constraints.get(index)
    }

    /// Get all constraints
    pub fn all(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Check if all constraints are satisfied
    pub fn all_satisfied<C>(&self, context: &C) -> Result<bool, ConstraintError>
    where
        C: crate::parameters::expression::EvaluationContext,
    {
        for constraint in &self.constraints {
            if !constraint.is_satisfied(context)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Get a list of violated constraints
    pub fn violated<C>(&self, context: &C) -> Result<Vec<&Constraint>, ConstraintError>
    where
        C: crate::parameters::expression::EvaluationContext,
    {
        let mut violated = Vec::new();

        for constraint in &self.constraints {
            if !constraint.is_satisfied(context)? {
                violated.push(constraint);
            }
        }

        Ok(violated)
    }

    /// Get all parameters that are used in constraints
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for constraint in &self.constraints {
            for var in constraint.variables() {
                if !names.contains(&var) {
                    names.push(var);
                }
            }
        }

        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::expression::SimpleContext;

    #[test]
    fn test_constraint_creation() {
        // Create a constraint
        let c = Constraint::new("x", ConstraintType::LessThan, "10").unwrap();
        assert_eq!(c.lhs, "x");
        assert_eq!(c.constraint_type, ConstraintType::LessThan);
        assert_eq!(c.rhs, "10");
        assert!(c.expr.is_some());

        // The constraint should be "x - 10", which is satisfied if x - 10 < 0
        assert_eq!(c.to_string(), "x < 10");

        // Check the variables
        let vars = c.variables();
        assert_eq!(vars.len(), 1);
        assert!(vars.contains(&"x".to_string()));
    }

    #[test]
    fn test_constraint_satisfaction() {
        // Create a constraint
        let c = Constraint::new("x", ConstraintType::LessThan, "10").unwrap();

        // Create a context for evaluation
        let mut context = SimpleContext::new();
        context.set_variable("x", 5.0);

        // Check if the constraint is satisfied
        assert!(c.is_satisfied(&context).unwrap());

        // Update the context
        context.set_variable("x", 15.0);

        // Check if the constraint is satisfied
        assert!(!c.is_satisfied(&context).unwrap());
    }

    #[test]
    fn test_constraint_types() {
        // Create a context
        let mut context = SimpleContext::new();
        context.set_variable("x", 5.0);
        context.set_variable("y", 10.0);

        // Equal
        let c_eq = Constraint::new("x", ConstraintType::Equal, "5").unwrap();
        assert!(c_eq.is_satisfied(&context).unwrap());

        // Not equal
        let c_ne = Constraint::new("x", ConstraintType::NotEqual, "10").unwrap();
        assert!(c_ne.is_satisfied(&context).unwrap());

        // Less than
        let c_lt = Constraint::new("x", ConstraintType::LessThan, "y").unwrap();
        assert!(c_lt.is_satisfied(&context).unwrap());

        // Less than or equal
        let c_le = Constraint::new("x", ConstraintType::LessThanOrEqual, "5").unwrap();
        assert!(c_le.is_satisfied(&context).unwrap());

        // Greater than
        let c_gt = Constraint::new("y", ConstraintType::GreaterThan, "x").unwrap();
        assert!(c_gt.is_satisfied(&context).unwrap());

        // Greater than or equal
        let c_ge = Constraint::new("y", ConstraintType::GreaterThanOrEqual, "10").unwrap();
        assert!(c_ge.is_satisfied(&context).unwrap());
    }

    #[test]
    fn test_complex_constraints() {
        // Create a context
        let mut context = SimpleContext::new();
        context.set_variable("x", 5.0);
        context.set_variable("y", 10.0);
        context.set_variable("z", 15.0);

        // Complex constraint: x + y < z
        let c1 = Constraint::new("x + y", ConstraintType::LessThan, "z").unwrap();
        let expr1 = c1.get_or_parse_expr().unwrap();
        let value1 = expr1.evaluate(&context).unwrap();
        println!(
            "c1: {} {} {}, expression: {:?}, value: {}",
            c1.lhs,
            c1.constraint_type.as_operator(),
            c1.rhs,
            expr1,
            value1
        );

        // For "x + y < z" with x=5, y=10, z=15, we have 5+10=15, which is equal to z (15)
        // So the constraint should be: (5+10) - 15 = 0, which is not < 0
        // Let's make a small adjustment to the test values
        context.set_variable("z", 16.0); // Now x + y (5+10=15) < z (16) should be true

        assert!(c1.is_satisfied(&context).unwrap());

        // Complex constraint: x^2 + y^2 < z^2
        let c2 = Constraint::new("x^2 + y^2", ConstraintType::LessThan, "z^2").unwrap();
        let expr2 = c2.get_or_parse_expr().unwrap();
        let value2 = expr2.evaluate(&context).unwrap();
        println!(
            "c2: {} {} {}, expression: {:?}, value: {}",
            c2.lhs,
            c2.constraint_type.as_operator(),
            c2.rhs,
            expr2,
            value2
        );
        assert!(c2.is_satisfied(&context).unwrap());

        // Complex constraint with functions: sin(x) + cos(y) < z
        let c3 = Constraint::new("sin(x) + cos(y)", ConstraintType::LessThan, "z").unwrap();
        let expr3 = c3.get_or_parse_expr().unwrap();
        let value3 = expr3.evaluate(&context).unwrap();
        println!(
            "c3: {} {} {}, expression: {:?}, value: {}",
            c3.lhs,
            c3.constraint_type.as_operator(),
            c3.rhs,
            expr3,
            value3
        );
        assert!(c3.is_satisfied(&context).unwrap());
    }

    #[test]
    fn test_constraints_collection() {
        // Create a constraints collection
        let mut constraints = Constraints::new();
        assert!(constraints.is_empty());

        // Add constraints
        constraints
            .add_constraint("x", ConstraintType::LessThan, "10")
            .unwrap();
        constraints
            .add_constraint("y", ConstraintType::GreaterThan, "5")
            .unwrap();
        constraints
            .add_constraint("x + y", ConstraintType::Equal, "z")
            .unwrap();

        assert_eq!(constraints.len(), 3);

        // Check the parameter names
        let names = constraints.parameter_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"x".to_string()));
        assert!(names.contains(&"y".to_string()));
        assert!(names.contains(&"z".to_string()));

        // Create a context for evaluation
        let mut context = SimpleContext::new();
        context.set_variable("x", 5.0);
        context.set_variable("y", 10.0);
        context.set_variable("z", 15.0);

        // Check if all constraints are satisfied
        assert!(constraints.all_satisfied(&context).unwrap());

        // Get violated constraints (should be none)
        let violated = constraints.violated(&context).unwrap();
        assert!(violated.is_empty());

        // Update the context to violate a constraint
        context.set_variable("x", 15.0);

        // Check if all constraints are satisfied
        assert!(!constraints.all_satisfied(&context).unwrap());

        // Get violated constraints
        let violated = constraints.violated(&context).unwrap();

        // Given our context: x=15, y=10, z=15
        // 1. x < 10: 15 < 10 (false)
        // 2. y > 5: 10 > 5 (true)
        // 3. x + y = z: 15 + 10 = 15 (false, since 25 != 15)

        // We should have 2 violated constraints: "x < 10" and "x + y = z"
        println!(
            "Violated constraints: {:?}",
            violated.iter().map(|c| c.to_string()).collect::<Vec<_>>()
        );

        assert_eq!(violated.len(), 2);

        // Just check if the first expected violation is in the list
        let has_x_lt_10 = violated.iter().any(|c| c.to_string() == "x < 10");
        assert!(has_x_lt_10, "Should have violated constraint 'x < 10'");

        // Remove the "x < 10" constraint
        let idx_to_remove = constraints
            .constraints
            .iter()
            .position(|c| c.to_string() == "x < 10")
            .expect("Should find the 'x < 10' constraint");
        let removed = constraints.remove(idx_to_remove).unwrap();
        assert_eq!(removed.to_string(), "x < 10");
        assert_eq!(constraints.len(), 2);

        // We still have the equality constraint that's violated
        // We need to update the context to make all remaining constraints satisfied
        context.set_variable("z", 25.0); // Now x + y (15+10=25) = z (25)

        // Now all constraints should be satisfied
        assert!(constraints.all_satisfied(&context).unwrap());
        assert!(constraints.all_satisfied(&context).unwrap());
    }
}
