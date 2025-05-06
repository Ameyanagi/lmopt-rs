//! Expression parsing and evaluation for parameter constraints
//!
//! This module provides the ability to parse and evaluate mathematical expressions
//! for parameter constraints, similar to lmfit-py's expression system.

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0},
    combinator::recognize,
    multi::many0,
    number::complete::double,
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};
use std::collections::HashMap;
use thiserror::Error;

/// Error that can occur during expression parsing or evaluation
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ExpressionError {
    #[error("Failed to parse expression: {message}")]
    ParseError { message: String },

    #[error("Undefined variable: {name}")]
    UndefinedVariable { name: String },

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Undefined function: {name}")]
    UndefinedFunction { name: String },
}

/// Result type for expression evaluation
type ExprResult<T> = Result<T, ExpressionError>;

/// Expression AST node
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Constant number
    Number(f64),

    /// Variable reference
    Variable(String),

    /// Unary operations
    Unary(UnaryOp, Box<Expression>),

    /// Binary operations
    Binary(BinaryOp, Box<Expression>, Box<Expression>),

    /// Function call
    Function(String, Vec<Expression>),
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    /// Negation (-)
    Neg,
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    /// Addition (+)
    Add,

    /// Subtraction (-)
    Sub,

    /// Multiplication (*)
    Mul,

    /// Division (/)
    Div,

    /// Power (^)
    Pow,
}

/// Context for expression evaluation, providing variable values
pub trait EvaluationContext {
    /// Get the value of a variable
    fn get_variable(&self, name: &str) -> ExprResult<f64>;

    /// Check if a variable exists
    fn has_variable(&self, name: &str) -> bool;

    /// Get the names of all variables
    fn variable_names(&self) -> Vec<String>;
}

/// Simple implementation of EvaluationContext using a HashMap
#[derive(Debug, Clone, Default)]
pub struct SimpleContext {
    /// Map of variable names to values
    variables: HashMap<String, f64>,
}

impl SimpleContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Set a variable value
    pub fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
    }

    /// Remove a variable
    pub fn remove_variable(&mut self, name: &str) -> Option<f64> {
        self.variables.remove(name)
    }

    /// Create a new context with the given variables
    pub fn with_variables(variables: HashMap<String, f64>) -> Self {
        Self { variables }
    }
}

impl EvaluationContext for SimpleContext {
    fn get_variable(&self, name: &str) -> ExprResult<f64> {
        self.variables
            .get(name)
            .copied()
            .ok_or_else(|| ExpressionError::UndefinedVariable {
                name: name.to_string(),
            })
    }

    fn has_variable(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }

    fn variable_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }
}

impl EvaluationContext for HashMap<String, f64> {
    fn get_variable(&self, name: &str) -> ExprResult<f64> {
        self.get(name)
            .copied()
            .ok_or_else(|| ExpressionError::UndefinedVariable {
                name: name.to_string(),
            })
    }

    fn has_variable(&self, name: &str) -> bool {
        self.contains_key(name)
    }

    fn variable_names(&self) -> Vec<String> {
        self.keys().cloned().collect()
    }
}

impl Expression {
    /// Parse an expression from a string
    pub fn parse(input: &str) -> ExprResult<Self> {
        match expr_parser(input.trim()) {
            Ok((remainder, expr)) => {
                // Make sure the entire input was consumed
                if remainder.trim().is_empty() {
                    Ok(expr)
                } else {
                    Err(ExpressionError::ParseError {
                        message: format!("Unexpected trailing characters: '{}'", remainder),
                    })
                }
            }
            Err(e) => Err(ExpressionError::ParseError {
                message: format!("{:?}", e),
            }),
        }
    }

    /// Evaluate the expression with the given context
    pub fn evaluate<C: EvaluationContext>(&self, context: &C) -> ExprResult<f64> {
        match self {
            Self::Number(n) => Ok(*n),

            Self::Variable(name) => context.get_variable(name),

            Self::Unary(op, expr) => {
                let value = expr.evaluate(context)?;
                match op {
                    UnaryOp::Neg => Ok(-value),
                }
            }

            Self::Binary(op, left, right) => {
                let lhs = left.evaluate(context)?;
                let rhs = right.evaluate(context)?;

                match op {
                    BinaryOp::Add => Ok(lhs + rhs),
                    BinaryOp::Sub => Ok(lhs - rhs),
                    BinaryOp::Mul => Ok(lhs * rhs),
                    BinaryOp::Div => {
                        if rhs == 0.0 {
                            Err(ExpressionError::DivisionByZero)
                        } else {
                            Ok(lhs / rhs)
                        }
                    }
                    BinaryOp::Pow => Ok(lhs.powf(rhs)),
                }
            }

            Self::Function(name, args) => {
                // Evaluate all arguments
                let mut evaluated_args = Vec::with_capacity(args.len());
                for arg in args {
                    evaluated_args.push(arg.evaluate(context)?);
                }

                // Call the appropriate function
                match name.as_str() {
                    "sin" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "sin() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].sin())
                    }
                    "cos" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "cos() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].cos())
                    }
                    "tan" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "tan() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].tan())
                    }
                    "exp" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "exp() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].exp())
                    }
                    "log" | "ln" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "log() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].ln())
                    }
                    "log10" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "log10() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].log10())
                    }
                    "sqrt" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "sqrt() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].sqrt())
                    }
                    "abs" => {
                        if evaluated_args.len() != 1 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "abs() requires 1 argument, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args[0].abs())
                    }
                    "max" => {
                        if evaluated_args.len() < 2 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "max() requires at least 2 arguments, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args
                            .iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
                    }
                    "min" => {
                        if evaluated_args.len() < 2 {
                            return Err(ExpressionError::InvalidOperation {
                                message: format!(
                                    "min() requires at least 2 arguments, got {}",
                                    evaluated_args.len()
                                ),
                            });
                        }
                        Ok(evaluated_args.iter().fold(f64::INFINITY, |a, &b| a.min(b)))
                    }
                    _ => Err(ExpressionError::UndefinedFunction {
                        name: name.to_string(),
                    }),
                }
            }
        }
    }

    /// Find all variable names used in the expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    /// Recursively collect all variable names used in the expression
    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Self::Number(_) => {}

            Self::Variable(name) => {
                vars.push(name.clone());
            }

            Self::Unary(_, expr) => {
                expr.collect_variables(vars);
            }

            Self::Binary(_, left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }

            Self::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
        }
    }
}

// Parser functions using nom

/// Parse an identifier (variable or function name)
fn identifier(input: &str) -> IResult<&str, String> {
    let mut parser = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ));

    let (input, matched) = parser.parse(input)?;
    Ok((input, matched.to_string()))
}

/// Parse a comma-separated list of expressions (for function arguments)
fn args_list(input: &str) -> IResult<&str, Vec<Expression>> {
    let (input, first) = expr_parser(input)?;
    let mut res = vec![first];

    let mut remainder = input;
    loop {
        let mut comma_parser = delimited(
            multispace0::<&str, nom::error::Error<&str>>,
            char::<&str, nom::error::Error<&str>>(','),
            multispace0::<&str, nom::error::Error<&str>>,
        );

        // Try to parse a comma
        match comma_parser.parse(remainder) {
            Ok((after_comma, _)) => {
                // Try to parse an expression after the comma
                match expr_parser(after_comma) {
                    Ok((after_expr, expr)) => {
                        res.push(expr);
                        remainder = after_expr;
                    }
                    Err(_) => break,
                }
            }
            Err(_) => break,
        }
    }

    Ok((remainder, res))
}

/// Parse a function call
fn function_call(input: &str) -> IResult<&str, Expression> {
    let (input, name) = identifier(input)?;
    let mut space_parser = multispace0::<&str, nom::error::Error<&str>>;
    let (input, _) = space_parser.parse(input)?;
    let mut open_paren_parser = char::<&str, nom::error::Error<&str>>('(');
    let (input, _) = open_paren_parser.parse(input)?;
    let mut space_parser2 = multispace0::<&str, nom::error::Error<&str>>;
    let (input, _) = space_parser2.parse(input)?;

    // Handle empty arguments case
    let mut close_paren_parser = char::<&str, nom::error::Error<&str>>(')');
    if let Ok((input, _)) = close_paren_parser.parse(input) {
        return Ok((input, Expression::Function(name, vec![])));
    }

    // Handle non-empty arguments case
    let (input, args) = args_list(input)?;
    let (input, _) = multispace0.parse(input)?;

    let mut close_paren_parser = char::<&str, nom::error::Error<&str>>(')');
    let (input, _) = close_paren_parser.parse(input)?;

    Ok((input, Expression::Function(name, args)))
}

/// Parse a number
fn number(input: &str) -> IResult<&str, Expression> {
    let (input, num) = double(input)?;
    Ok((input, Expression::Number(num)))
}

/// Parse a variable reference
fn variable(input: &str) -> IResult<&str, Expression> {
    let (input, var_name) = identifier(input)?;
    Ok((input, Expression::Variable(var_name)))
}

/// Parse a parenthesized expression
fn parens(input: &str) -> IResult<&str, Expression> {
    let (input, _) = char('(').parse(input)?;
    let (input, _) = multispace0.parse(input)?;
    let (input, expr) = expr_parser(input)?;
    let (input, _) = multispace0.parse(input)?;
    let (input, _) = char::<_, nom::error::Error<_>>(')').parse(input)?;
    Ok((input, expr))
}

/// Parse a primary expression (number, variable, function call, or parenthesized expression)
fn primary(input: &str) -> IResult<&str, Expression> {
    let number_result = number(input);
    if let Ok(result) = number_result {
        return Ok(result);
    }

    let function_result = function_call(input);
    if let Ok(result) = function_result {
        return Ok(result);
    }

    let variable_result = variable(input);
    if let Ok(result) = variable_result {
        return Ok(result);
    }

    parens(input)
}

/// Parse a unary expression (-expr)
fn unary(input: &str) -> IResult<&str, Expression> {
    let (input, _) = multispace0.parse(input)?;

    // Try to parse a negative expression
    let mut neg_parser = preceded(char('-'), primary);
    match neg_parser.parse(input) {
        Ok((remaining, expr)) => {
            return Ok((remaining, Expression::Unary(UnaryOp::Neg, Box::new(expr))));
        }
        Err(_) => {
            // Otherwise, parse a primary expression
            primary(input)
        }
    }
}

/// Parse a power expression (expr ^ expr)
fn power(input: &str) -> IResult<&str, Expression> {
    let (input, left) = unary(input)?;
    let (input, _) = multispace0.parse(input)?;

    let mut op_parser = char::<_, nom::error::Error<_>>('^');
    let op_parser_result = op_parser.parse(input);
    match op_parser_result {
        Ok((after_op, _)) => {
            let (after_op, _) = multispace0.parse(after_op)?;
            let (after_right, right) = power(after_op)?;
            Ok((
                after_right,
                Expression::Binary(BinaryOp::Pow, Box::new(left), Box::new(right)),
            ))
        }
        Err(_) => Ok((input, left)),
    }
}

/// Parse a multiplicative expression (expr * expr, expr / expr)
fn term(input: &str) -> IResult<&str, Expression> {
    let (input, left) = power(input)?;
    let (input, _) = multispace0.parse(input)?;

    // Try to match * or /
    let mut mul_op_parser = char::<_, nom::error::Error<_>>('*');
    let mut div_op_parser = char::<_, nom::error::Error<_>>('/');
    let mul_op_result = mul_op_parser.parse(input);
    let div_op_result = div_op_parser.parse(input);

    match (mul_op_result, div_op_result) {
        (Ok((after_op, _)), _) => {
            // Multiply operation
            let (after_op, _) = multispace0.parse(after_op)?;
            let (remaining, right) = term(after_op)?;
            Ok((
                remaining,
                Expression::Binary(BinaryOp::Mul, Box::new(left), Box::new(right)),
            ))
        }
        (_, Ok((after_op, _))) => {
            // Divide operation
            let (after_op, _) = multispace0.parse(after_op)?;
            let (remaining, right) = term(after_op)?;
            Ok((
                remaining,
                Expression::Binary(BinaryOp::Div, Box::new(left), Box::new(right)),
            ))
        }
        _ => {
            // No operator, return left expression
            Ok((input, left))
        }
    }
}

/// Parse an additive expression (expr + expr, expr - expr)
fn expr_parser(input: &str) -> IResult<&str, Expression> {
    let (input, _) = multispace0.parse(input)?;
    let (input, left) = term(input)?;
    let (input, _) = multispace0.parse(input)?;

    // Try to match + or -
    let mut add_op_parser = char::<_, nom::error::Error<_>>('+');
    let mut sub_op_parser = char::<_, nom::error::Error<_>>('-');
    let add_op_result = add_op_parser.parse(input);
    let sub_op_result = sub_op_parser.parse(input);

    match (add_op_result, sub_op_result) {
        (Ok((after_op, _)), _) => {
            // Add operation
            let (after_op, _) = multispace0.parse(after_op)?;
            let (remaining, right) = expr_parser(after_op)?;
            Ok((
                remaining,
                Expression::Binary(BinaryOp::Add, Box::new(left), Box::new(right)),
            ))
        }
        (_, Ok((after_op, _))) => {
            // Subtract operation
            let (after_op, _) = multispace0.parse(after_op)?;
            let (remaining, right) = expr_parser(after_op)?;
            Ok((
                remaining,
                Expression::Binary(BinaryOp::Sub, Box::new(left), Box::new(right)),
            ))
        }
        _ => {
            // No operator, return left expression
            Ok((input, left))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_number() {
        assert_eq!(Expression::parse("42").unwrap(), Expression::Number(42.0));

        assert_eq!(Expression::parse("3.14").unwrap(), Expression::Number(3.14));

        assert_eq!(
            Expression::parse("-2.5").unwrap(),
            Expression::Unary(UnaryOp::Neg, Box::new(Expression::Number(2.5)))
        );
    }

    #[test]
    fn test_parse_variable() {
        assert_eq!(
            Expression::parse("x").unwrap(),
            Expression::Variable("x".to_string())
        );

        assert_eq!(
            Expression::parse("variable_name").unwrap(),
            Expression::Variable("variable_name".to_string())
        );

        assert_eq!(
            Expression::parse("var_1").unwrap(),
            Expression::Variable("var_1".to_string())
        );
    }

    #[test]
    fn test_parse_binary_ops() {
        assert_eq!(
            Expression::parse("1 + 2").unwrap(),
            Expression::Binary(
                BinaryOp::Add,
                Box::new(Expression::Number(1.0)),
                Box::new(Expression::Number(2.0))
            )
        );

        assert_eq!(
            Expression::parse("3 - 4").unwrap(),
            Expression::Binary(
                BinaryOp::Sub,
                Box::new(Expression::Number(3.0)),
                Box::new(Expression::Number(4.0))
            )
        );

        assert_eq!(
            Expression::parse("5 * 6").unwrap(),
            Expression::Binary(
                BinaryOp::Mul,
                Box::new(Expression::Number(5.0)),
                Box::new(Expression::Number(6.0))
            )
        );

        assert_eq!(
            Expression::parse("7 / 8").unwrap(),
            Expression::Binary(
                BinaryOp::Div,
                Box::new(Expression::Number(7.0)),
                Box::new(Expression::Number(8.0))
            )
        );

        assert_eq!(
            Expression::parse("2 ^ 3").unwrap(),
            Expression::Binary(
                BinaryOp::Pow,
                Box::new(Expression::Number(2.0)),
                Box::new(Expression::Number(3.0))
            )
        );
    }

    #[test]
    fn test_parse_complex_expression() {
        // Get the actual expression tree
        let expr = Expression::parse("2 * (x + 1) / (4 - y)").unwrap();

        // Create a simple context to evaluate the expression
        let mut context = SimpleContext::new();
        context.set_variable("x", 2.0);
        context.set_variable("y", 3.0);

        // Test that the expression evaluates correctly
        assert_eq!(expr.evaluate(&context).unwrap(), 6.0);

        // Test that all expected variables are present
        let vars = expr.variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
    }

    #[test]
    fn test_parse_function_call() {
        assert_eq!(
            Expression::parse("sin(x)").unwrap(),
            Expression::Function(
                "sin".to_string(),
                vec![Expression::Variable("x".to_string())]
            )
        );

        assert_eq!(
            Expression::parse("max(x, y, 5)").unwrap(),
            Expression::Function(
                "max".to_string(),
                vec![
                    Expression::Variable("x".to_string()),
                    Expression::Variable("y".to_string()),
                    Expression::Number(5.0)
                ]
            )
        );
    }

    #[test]
    fn test_evaluate_simple() {
        let mut context = SimpleContext::new();
        context.set_variable("x", 2.0);
        context.set_variable("y", 3.0);

        // Test simple expressions
        assert_eq!(
            Expression::parse("42").unwrap().evaluate(&context).unwrap(),
            42.0
        );

        assert_eq!(
            Expression::parse("x").unwrap().evaluate(&context).unwrap(),
            2.0
        );

        assert_eq!(
            Expression::parse("-y").unwrap().evaluate(&context).unwrap(),
            -3.0
        );

        // Test binary operations
        assert_eq!(
            Expression::parse("x + y")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            5.0
        );

        assert_eq!(
            Expression::parse("x - y")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            -1.0
        );

        assert_eq!(
            Expression::parse("x * y")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            6.0
        );

        assert_eq!(
            Expression::parse("y / x")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            1.5
        );

        assert_eq!(
            Expression::parse("x ^ y")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            8.0
        );
    }

    #[test]
    fn test_evaluate_complex() {
        let mut context = SimpleContext::new();
        context.set_variable("x", 2.0);
        context.set_variable("y", 3.0);

        // Test complex expression
        assert_eq!(
            Expression::parse("2 * (x + 1) / (4 - y)")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            6.0
        );

        // Test function calls
        assert_eq!(
            Expression::parse("sin(x)")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            2.0_f64.sin()
        );

        assert_eq!(
            Expression::parse("max(x, y, 5)")
                .unwrap()
                .evaluate(&context)
                .unwrap(),
            5.0
        );
    }

    #[test]
    fn test_evaluation_errors() {
        let context = SimpleContext::new();

        // Test undefined variable
        match Expression::parse("x").unwrap().evaluate(&context) {
            Err(ExpressionError::UndefinedVariable { name }) => assert_eq!(name, "x"),
            _ => panic!("Expected UndefinedVariable error"),
        }

        // Test division by zero
        match Expression::parse("1 / 0").unwrap().evaluate(&context) {
            Err(ExpressionError::DivisionByZero) => {}
            _ => panic!("Expected DivisionByZero error"),
        }

        // Test undefined function
        match Expression::parse("foo(1)").unwrap().evaluate(&context) {
            Err(ExpressionError::UndefinedFunction { name }) => assert_eq!(name, "foo"),
            _ => panic!("Expected UndefinedFunction error"),
        }

        // Test function with wrong number of arguments
        match Expression::parse("sin(1, 2)").unwrap().evaluate(&context) {
            Err(ExpressionError::InvalidOperation { .. }) => {}
            _ => panic!("Expected InvalidOperation error"),
        }
    }

    #[test]
    fn test_variables() {
        // Test variable collection
        assert_eq!(
            Expression::parse("x + y * z").unwrap().variables(),
            vec!["x".to_string(), "y".to_string(), "z".to_string()]
        );

        assert_eq!(
            Expression::parse("sin(x) + cos(y)").unwrap().variables(),
            vec!["x".to_string(), "y".to_string()]
        );

        assert_eq!(
            Expression::parse("2 * (a + b) / (c - d)")
                .unwrap()
                .variables(),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }
}
