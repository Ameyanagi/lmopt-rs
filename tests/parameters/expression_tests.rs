//! Tests for the Expression parsing and evaluation

use lmopt_rs::parameters::expression::{
    EvaluationContext, Expression, ExpressionError, SimpleContext,
};
use std::collections::HashMap;

#[test]
fn test_expression_parsing() {
    // Test parsing simple expressions
    let expr1 = Expression::parse("42").unwrap();
    assert!(expr1.variables().is_empty());

    let expr2 = Expression::parse("x").unwrap();
    assert_eq!(expr2.variables().len(), 1);
    assert!(expr2.variables().contains(&"x".to_string()));

    let expr3 = Expression::parse("x + y").unwrap();
    assert_eq!(expr3.variables().len(), 2);
    assert!(expr3.variables().contains(&"x".to_string()));
    assert!(expr3.variables().contains(&"y".to_string()));

    let expr4 = Expression::parse("x * y").unwrap();
    assert_eq!(expr4.variables().len(), 2);

    let expr5 = Expression::parse("x - y").unwrap();
    assert_eq!(expr5.variables().len(), 2);

    let expr6 = Expression::parse("x / y").unwrap();
    assert_eq!(expr6.variables().len(), 2);

    let expr7 = Expression::parse("x^2").unwrap();
    assert_eq!(expr7.variables().len(), 1);

    // Test parsing complex expressions
    let expr8 = Expression::parse("2 * (x + 1)").unwrap();
    assert_eq!(expr8.variables().len(), 1);

    let expr9 = Expression::parse("sin(x) + cos(y)").unwrap();
    assert_eq!(expr9.variables().len(), 2);

    let expr10 = Expression::parse("max(a, b, c)").unwrap();
    assert_eq!(expr10.variables().len(), 3);

    let expr11 = Expression::parse("(x + y) * (z - 1) / w").unwrap();
    assert_eq!(expr11.variables().len(), 4);

    // Test parsing expressions with whitespace
    let expr12 = Expression::parse(" x + y ").unwrap();
    assert_eq!(expr12.variables().len(), 2);

    let expr13 = Expression::parse("x   +   y").unwrap();
    assert_eq!(expr13.variables().len(), 2);

    let expr14 = Expression::parse("  (  x  +  y  )  *  z  ").unwrap();
    assert_eq!(expr14.variables().len(), 3);

    // Test parsing expressions with negative numbers
    let expr15 = Expression::parse("-x").unwrap();
    assert_eq!(expr15.variables().len(), 1);

    let expr16 = Expression::parse("x + (-y)").unwrap();
    assert_eq!(expr16.variables().len(), 2);

    let expr17 = Expression::parse("-2 * x").unwrap();
    assert_eq!(expr17.variables().len(), 1);

    // Test parsing invalid expressions
    let result = Expression::parse("");
    assert!(result.is_err());

    let result = Expression::parse("x +");
    assert!(result.is_err());

    let result = Expression::parse("x + (y");
    assert!(result.is_err());

    let result = Expression::parse("@#$%");
    assert!(result.is_err());
}

#[test]
fn test_expression_variables() {
    // Test variable extraction from expressions
    let expr = Expression::parse("x + y").unwrap();
    let vars = expr.variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));

    let expr = Expression::parse("x + x + x").unwrap();
    let vars = expr.variables();
    assert_eq!(vars.len(), 1);
    assert!(vars.contains(&"x".to_string()));

    let expr = Expression::parse("2 * (x + y) / z").unwrap();
    let vars = expr.variables();
    assert_eq!(vars.len(), 3);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));
    assert!(vars.contains(&"z".to_string()));

    let expr = Expression::parse("sin(x) + cos(y)").unwrap();
    let vars = expr.variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));

    let expr = Expression::parse("42").unwrap();
    let vars = expr.variables();
    assert_eq!(vars.len(), 0);
}

#[test]
fn test_expression_evaluation_simple() {
    // Create a context with variables
    let mut context = SimpleContext::new();
    context.set_variable("x", 2.0);
    context.set_variable("y", 3.0);

    // Test evaluating simple expressions
    let expr = Expression::parse("42").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 42.0);

    let expr = Expression::parse("x").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 2.0);

    let expr = Expression::parse("y").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 3.0);

    let expr = Expression::parse("-x").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), -2.0);

    // Test evaluating basic operations
    let expr = Expression::parse("x + y").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 5.0);

    let expr = Expression::parse("x - y").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), -1.0);

    let expr = Expression::parse("x * y").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 6.0);

    let expr = Expression::parse("y / x").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 1.5);

    let expr = Expression::parse("x^2").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 4.0);

    // Test evaluating complex expressions
    let expr = Expression::parse("2 * (x + y)").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 10.0);

    let expr = Expression::parse("(x + y) * (x - y)").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), -5.0);

    let expr = Expression::parse("x^2 + y^2").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 13.0);
}

#[test]
fn test_expression_evaluation_functions() {
    // Create a context with variables
    let mut context = SimpleContext::new();
    context.set_variable("x", 2.0);
    context.set_variable("y", 3.0);
    context.set_variable("z", 4.0);

    // Test evaluating mathematical functions
    let expr = Expression::parse("sin(x)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::sin(2.0)).abs() < 1e-10);

    let expr = Expression::parse("cos(y)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::cos(3.0)).abs() < 1e-10);

    let expr = Expression::parse("tan(x)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::tan(2.0)).abs() < 1e-10);

    let expr = Expression::parse("exp(x)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::exp(2.0)).abs() < 1e-10);

    let expr = Expression::parse("log(y)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::ln(3.0)).abs() < 1e-10);

    let expr = Expression::parse("ln(y)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::ln(3.0)).abs() < 1e-10);

    let expr = Expression::parse("log10(z)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - f64::log10(4.0)).abs() < 1e-10);

    let expr = Expression::parse("sqrt(z)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - 2.0).abs() < 1e-10);

    let expr = Expression::parse("abs(-x)").unwrap();
    assert!((expr.evaluate(&context).unwrap() - 2.0).abs() < 1e-10);

    // Test evaluating functions with multiple arguments
    let expr = Expression::parse("max(x, y, z)").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 4.0);

    let expr = Expression::parse("min(x, y, z)").unwrap();
    assert_eq!(expr.evaluate(&context).unwrap(), 2.0);

    // Test evaluating complex expressions with functions
    let expr = Expression::parse("sin(x)^2 + cos(x)^2").unwrap();
    assert!((expr.evaluate(&context).unwrap() - 1.0).abs() < 1e-10);

    let expr = Expression::parse("log(exp(x))").unwrap();
    assert!((expr.evaluate(&context).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_expression_evaluation_errors() {
    // Create a context with variables
    let mut context = SimpleContext::new();
    context.set_variable("x", 2.0);

    // Test undefined variable
    let expr = Expression::parse("y").unwrap();
    let result = expr.evaluate(&context);
    assert!(result.is_err());
    match result {
        Err(ExpressionError::UndefinedVariable { name }) => assert_eq!(name, "y"),
        _ => panic!("Expected UndefinedVariable error"),
    }

    // Test division by zero
    let expr = Expression::parse("x / 0").unwrap();
    let result = expr.evaluate(&context);
    assert!(result.is_err());
    match result {
        Err(ExpressionError::DivisionByZero) => {}
        _ => panic!("Expected DivisionByZero error"),
    }

    // Test undefined function
    let expr = Expression::parse("unknown_func(x)").unwrap();
    let result = expr.evaluate(&context);
    assert!(result.is_err());
    match result {
        Err(ExpressionError::UndefinedFunction { name }) => assert_eq!(name, "unknown_func"),
        _ => panic!("Expected UndefinedFunction error"),
    }

    // Test invalid function arguments
    let expr = Expression::parse("sin(x, y)").unwrap();
    let result = expr.evaluate(&context);
    assert!(result.is_err());
    match result {
        Err(ExpressionError::InvalidOperation { .. }) => {}
        _ => panic!("Expected InvalidOperation error"),
    }

    let expr = Expression::parse("max(x)").unwrap();
    let result = expr.evaluate(&context);
    assert!(result.is_err());
    match result {
        Err(ExpressionError::InvalidOperation { .. }) => {}
        _ => panic!("Expected InvalidOperation error"),
    }
}

#[test]
fn test_expression_context_implementations() {
    // Test SimpleContext
    let mut context = SimpleContext::new();
    context.set_variable("x", 2.0);
    context.set_variable("y", 3.0);

    assert!(context.has_variable("x"));
    assert!(context.has_variable("y"));
    assert!(!context.has_variable("z"));

    assert_eq!(context.get_variable("x").unwrap(), 2.0);
    assert_eq!(context.get_variable("y").unwrap(), 3.0);
    assert!(context.get_variable("z").is_err());

    let names = context.variable_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"x".to_string()));
    assert!(names.contains(&"y".to_string()));

    // Test HashMap implementation
    let mut hash_map = HashMap::new();
    hash_map.insert("a".to_string(), 4.0);
    hash_map.insert("b".to_string(), 5.0);

    assert!(hash_map.has_variable("a"));
    assert!(hash_map.has_variable("b"));
    assert!(!hash_map.has_variable("c"));

    assert_eq!(hash_map.get_variable("a").unwrap(), 4.0);
    assert_eq!(hash_map.get_variable("b").unwrap(), 5.0);
    assert!(hash_map.get_variable("c").is_err());

    let names = hash_map.variable_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
}
