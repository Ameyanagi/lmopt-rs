//! Tests for the Constraints implementation

use lmopt_rs::parameters::constraints::{Constraint, ConstraintType, Constraints};
use lmopt_rs::parameters::expression::SimpleContext;

#[test]
fn test_constraint_creation() {
    // Test creating basic constraints
    let c = Constraint::new("x", ConstraintType::LessThan, "10").unwrap();
    assert_eq!(c.lhs, "x");
    assert_eq!(c.constraint_type, ConstraintType::LessThan);
    assert_eq!(c.rhs, "10");
    assert_eq!(c.to_string(), "x < 10");

    let c = Constraint::new("y", ConstraintType::GreaterThan, "0").unwrap();
    assert_eq!(c.lhs, "y");
    assert_eq!(c.constraint_type, ConstraintType::GreaterThan);
    assert_eq!(c.rhs, "0");
    assert_eq!(c.to_string(), "y > 0");

    let c = Constraint::new("a", ConstraintType::Equal, "b").unwrap();
    assert_eq!(c.lhs, "a");
    assert_eq!(c.constraint_type, ConstraintType::Equal);
    assert_eq!(c.rhs, "b");
    assert_eq!(c.to_string(), "a == b");

    // Test creating constraints with expressions
    let c = Constraint::new("x + y", ConstraintType::LessThan, "z").unwrap();
    assert_eq!(c.lhs, "x + y");
    assert_eq!(c.constraint_type, ConstraintType::LessThan);
    assert_eq!(c.rhs, "z");
    assert_eq!(c.to_string(), "x + y < z");

    let c = Constraint::new("sin(x)", ConstraintType::GreaterThanOrEqual, "cos(y)").unwrap();
    assert_eq!(c.lhs, "sin(x)");
    assert_eq!(c.constraint_type, ConstraintType::GreaterThanOrEqual);
    assert_eq!(c.rhs, "cos(y)");
    assert_eq!(c.to_string(), "sin(x) >= cos(y)");
}

#[test]
fn test_constraint_variables() {
    // Test getting variables used in constraints
    let c = Constraint::new("x", ConstraintType::LessThan, "10").unwrap();
    let vars = c.variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains(&"x".to_string()));

    let c = Constraint::new("x + y", ConstraintType::LessThan, "z").unwrap();
    let vars = c.variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));
    assert!(vars.contains(&"z".to_string()));

    let c = Constraint::new("sin(x) + cos(y)", ConstraintType::LessThan, "z * w").unwrap();
    let vars = c.variables();
    assert_eq!(vars.len(), 4);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));
    assert!(vars.contains(&"z".to_string()));
    assert!(vars.contains(&"w".to_string()));
}

#[test]
fn test_constraint_satisfaction() {
    // Create a context for evaluation
    let mut context = SimpleContext::new();
    context.set_variable("x", 5.0);
    context.set_variable("y", 10.0);
    context.set_variable("z", 15.0);

    // Test Less Than
    let c = Constraint::new("x", ConstraintType::LessThan, "10").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x", ConstraintType::LessThan, "5").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());

    // Test Greater Than
    let c = Constraint::new("y", ConstraintType::GreaterThan, "5").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("y", ConstraintType::GreaterThan, "10").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());

    // Test Equal
    let c = Constraint::new("x", ConstraintType::Equal, "5").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x", ConstraintType::Equal, "6").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());

    // Test Not Equal
    let c = Constraint::new("x", ConstraintType::NotEqual, "6").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x", ConstraintType::NotEqual, "5").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());

    // Test Less Than or Equal
    let c = Constraint::new("x", ConstraintType::LessThanOrEqual, "5").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x", ConstraintType::LessThanOrEqual, "4").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());

    // Test Greater Than or Equal
    let c = Constraint::new("y", ConstraintType::GreaterThanOrEqual, "10").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("y", ConstraintType::GreaterThanOrEqual, "11").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());
}

#[test]
fn test_constraint_expressions() {
    // Create a context for evaluation
    let mut context = SimpleContext::new();
    context.set_variable("x", 5.0);
    context.set_variable("y", 10.0);
    context.set_variable("z", 15.0);

    // Test complex expression constraints
    let c = Constraint::new("x + y", ConstraintType::LessThan, "z").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x * y", ConstraintType::LessThan, "z * 5").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("sin(x) + cos(y)", ConstraintType::LessThan, "1").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x^2", ConstraintType::LessThan, "y^2").unwrap();
    assert!(c.is_satisfied(&context).unwrap());

    let c = Constraint::new("x", ConstraintType::LessThan, "y/2").unwrap();
    assert!(!c.is_satisfied(&context).unwrap());
}

#[test]
fn test_constraints_collection() {
    // Create a constraints collection
    let mut constraints = Constraints::new();
    assert!(constraints.is_empty());
    assert_eq!(constraints.len(), 0);

    // Add a constraint
    constraints
        .add_constraint("x", ConstraintType::LessThan, "10")
        .unwrap();
    assert!(!constraints.is_empty());
    assert_eq!(constraints.len(), 1);

    // Add more constraints
    constraints
        .add_constraint("y", ConstraintType::GreaterThan, "0")
        .unwrap();
    constraints
        .add_constraint("x + y", ConstraintType::Equal, "z")
        .unwrap();
    assert_eq!(constraints.len(), 3);

    // Get constraints
    let c = constraints.get(0).unwrap();
    assert_eq!(c.to_string(), "x < 10");

    let c = constraints.get(1).unwrap();
    assert_eq!(c.to_string(), "y > 0");

    let c = constraints.get(2).unwrap();
    assert_eq!(c.to_string(), "x + y == z");

    // Get all constraints
    let all = constraints.all();
    assert_eq!(all.len(), 3);

    // Get parameter names
    let names = constraints.parameter_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"x".to_string()));
    assert!(names.contains(&"y".to_string()));
    assert!(names.contains(&"z".to_string()));

    // Remove a constraint
    let removed = constraints.remove(1).unwrap();
    assert_eq!(removed.to_string(), "y > 0");
    assert_eq!(constraints.len(), 2);

    // Get constraint that doesn't exist
    assert!(constraints.get(2).is_none());

    // Remove constraint that doesn't exist
    assert!(constraints.remove(5).is_none());
}

#[test]
fn test_constraints_satisfaction() {
    // Create a constraints collection
    let mut constraints = Constraints::new();
    constraints
        .add_constraint("x", ConstraintType::LessThan, "10")
        .unwrap();
    constraints
        .add_constraint("y", ConstraintType::GreaterThan, "0")
        .unwrap();
    constraints
        .add_constraint("x + y", ConstraintType::Equal, "z")
        .unwrap();

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

    // Update context to violate a constraint
    context.set_variable("z", 20.0);

    // Check if all constraints are satisfied
    assert!(!constraints.all_satisfied(&context).unwrap());

    // Get violated constraints
    let violated = constraints.violated(&context).unwrap();
    assert_eq!(violated.len(), 1);
    assert_eq!(violated[0].to_string(), "x + y == z");
}

#[test]
fn test_constraint_operator_display() {
    // Test operator string representation
    assert_eq!(ConstraintType::Equal.as_operator(), "==");
    assert_eq!(ConstraintType::NotEqual.as_operator(), "!=");
    assert_eq!(ConstraintType::LessThan.as_operator(), "<");
    assert_eq!(ConstraintType::LessThanOrEqual.as_operator(), "<=");
    assert_eq!(ConstraintType::GreaterThan.as_operator(), ">");
    assert_eq!(ConstraintType::GreaterThanOrEqual.as_operator(), ">=");
}
