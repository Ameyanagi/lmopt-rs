use std::collections::{HashMap, HashSet};
use std::fmt;

// Simple error types
#[derive(Debug)]
pub enum Error {
    ConstraintError(String),
    ExpressionError(String),
    ParameterError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConstraintError(msg) => write!(f, "Constraint error: {}", msg),
            Self::ExpressionError(msg) => write!(f, "Expression error: {}", msg),
            Self::ParameterError(msg) => write!(f, "Parameter error: {}", msg),
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

// Parameter implementation
#[derive(Debug, Clone)]
struct Parameter {
    name: String,
    value: f64,
    vary: bool,
    expr: Option<String>,
}

impl Parameter {
    fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            vary: true,
            expr: None,
        }
    }

    fn with_expr(name: &str, value: f64, expr: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            vary: false,
            expr: Some(expr.to_string()),
        }
    }

    fn value(&self) -> f64 {
        self.value
    }

    fn expr(&self) -> Option<&str> {
        self.expr.as_deref()
    }

    fn set_value(&mut self, value: f64) -> Result<()> {
        self.value = value;
        Ok(())
    }
}

// Simple expression
#[derive(Debug, Clone)]
struct Expression {
    expr: String,
}

impl Expression {
    fn parse(expr: &str) -> Result<Self> {
        Ok(Self {
            expr: expr.to_string(),
        })
    }

    fn evaluate(&self, context: &dyn EvaluationContext) -> Result<f64> {
        // Simple eval for testing only
        match self.expr.as_str() {
            // Basic variable reference
            "x" => context
                .get_variable("x")
                .map_err(|_| Error::ExpressionError("Variable x not found".to_string())),
            "y" => context
                .get_variable("y")
                .map_err(|_| Error::ExpressionError("Variable y not found".to_string())),

            // Operations
            "x - y" => {
                let x = context
                    .get_variable("x")
                    .map_err(|_| Error::ExpressionError("Variable x not found".to_string()))?;
                let y = context
                    .get_variable("y")
                    .map_err(|_| Error::ExpressionError("Variable y not found".to_string()))?;
                Ok(x - y)
            }
            "y - x" => {
                let x = context
                    .get_variable("x")
                    .map_err(|_| Error::ExpressionError("Variable x not found".to_string()))?;
                let y = context
                    .get_variable("y")
                    .map_err(|_| Error::ExpressionError("Variable y not found".to_string()))?;
                Ok(y - x)
            }
            "x + y" => {
                let x = context
                    .get_variable("x")
                    .map_err(|_| Error::ExpressionError("Variable x not found".to_string()))?;
                let y = context
                    .get_variable("y")
                    .map_err(|_| Error::ExpressionError("Variable y not found".to_string()))?;
                Ok(x + y)
            }

            // Numeric literals
            _ if self.expr.parse::<f64>().is_ok() => Ok(self.expr.parse::<f64>().unwrap()),

            // Anything else
            _ => Err(Error::ExpressionError(format!(
                "Unsupported expression: {}",
                self.expr
            ))),
        }
    }

    fn variables(&self) -> Vec<String> {
        match self.expr.as_str() {
            "x" => vec!["x".to_string()],
            "y" => vec!["y".to_string()],
            "x - y" | "y - x" | "x + y" => vec!["x".to_string(), "y".to_string()],
            _ => Vec::new(),
        }
    }
}

// Context for evaluating expressions
trait EvaluationContext {
    fn get_variable(&self, name: &str) -> Result<f64>;
    fn has_variable(&self, name: &str) -> bool;
}

// Constraint types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintType {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl ConstraintType {
    fn as_operator(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
        }
    }

    fn is_satisfied(&self, value: f64) -> bool {
        match self {
            Self::Equal => value.abs() < 1e-10,
            Self::NotEqual => value.abs() >= 1e-10,
            Self::LessThan => value < 0.0,
            Self::LessThanOrEqual => value <= 0.0,
            Self::GreaterThan => value > 0.0,
            Self::GreaterThanOrEqual => value >= 0.0,
        }
    }
}

// Constraint between parameters
#[derive(Debug, Clone)]
struct Constraint {
    lhs: String,
    constraint_type: ConstraintType,
    rhs: String,
    expr: Option<Expression>,
}

impl Constraint {
    fn new(lhs: &str, constraint_type: ConstraintType, rhs: &str) -> Result<Self> {
        // Create a combined expression based on the constraint type
        let combined_expr = match constraint_type {
            ConstraintType::LessThan | ConstraintType::LessThanOrEqual => {
                format!("{} - {}", lhs, rhs)
            }
            ConstraintType::GreaterThan | ConstraintType::GreaterThanOrEqual => {
                format!("{} - {}", rhs, lhs)
            }
            _ => format!("{} - {}", lhs, rhs),
        };

        let expr = Expression::parse(&combined_expr)?;

        Ok(Self {
            lhs: lhs.to_string(),
            constraint_type,
            rhs: rhs.to_string(),
            expr: Some(expr),
        })
    }

    fn is_satisfied(&self, context: &dyn EvaluationContext) -> Result<bool> {
        if let Some(expr) = &self.expr {
            let value = expr.evaluate(context)?;
            Ok(self.constraint_type.is_satisfied(value))
        } else {
            Err(Error::ConstraintError(
                "Expression not initialized".to_string(),
            ))
        }
    }

    fn to_string(&self) -> String {
        format!(
            "{} {} {}",
            self.lhs,
            self.constraint_type.as_operator(),
            self.rhs
        )
    }

    fn variables(&self) -> Vec<String> {
        if let Some(expr) = &self.expr {
            expr.variables()
        } else {
            Vec::new()
        }
    }
}

// Collection of constraints
#[derive(Debug, Clone)]
struct Constraints {
    constraints: Vec<Constraint>,
}

impl Constraints {
    fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    fn add(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    fn add_constraint(
        &mut self,
        lhs: &str,
        constraint_type: ConstraintType,
        rhs: &str,
    ) -> Result<()> {
        let constraint = Constraint::new(lhs, constraint_type, rhs)?;
        self.add(constraint);
        Ok(())
    }

    fn len(&self) -> usize {
        self.constraints.len()
    }

    fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    fn get(&self, index: usize) -> Option<&Constraint> {
        self.constraints.get(index)
    }

    fn all(&self) -> &[Constraint] {
        &self.constraints
    }

    fn all_satisfied(&self, context: &dyn EvaluationContext) -> Result<bool> {
        for constraint in &self.constraints {
            if !constraint.is_satisfied(context)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn violated<'a>(&'a self, context: &dyn EvaluationContext) -> Result<Vec<&'a Constraint>> {
        let mut violated = Vec::new();

        for constraint in &self.constraints {
            if !constraint.is_satisfied(context)? {
                violated.push(constraint);
            }
        }

        Ok(violated)
    }
}

// Parameters collection
#[derive(Debug, Clone)]
struct Parameters {
    params: HashMap<String, Parameter>,
    deps: HashMap<String, HashSet<String>>,
    constraints: Constraints,
}

impl Parameters {
    fn new() -> Self {
        Self {
            params: HashMap::new(),
            deps: HashMap::new(),
            constraints: Constraints::new(),
        }
    }

    fn add_param(&mut self, name: &str, value: f64) -> Result<()> {
        let param = Parameter::new(name, value);
        self.params.insert(name.to_string(), param);
        self.deps
            .entry(name.to_string())
            .or_insert_with(HashSet::new);
        Ok(())
    }

    fn add_param_with_expr(&mut self, name: &str, value: f64, expr: &str) -> Result<()> {
        let param = Parameter::with_expr(name, value, expr);
        self.params.insert(name.to_string(), param);
        self.deps
            .entry(name.to_string())
            .or_insert_with(HashSet::new);
        self.update_deps()?;
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&Parameter> {
        self.params.get(name)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.params.get_mut(name)
    }

    fn update_deps(&mut self) -> Result<()> {
        // Clear existing dependencies
        for deps in self.deps.values_mut() {
            deps.clear();
        }

        // Build dependencies based on expressions
        for (name, param) in &self.params {
            if let Some(expr_str) = param.expr() {
                let expr = Expression::parse(expr_str)?;

                for var_name in expr.variables() {
                    if self.params.contains_key(&var_name) && var_name != *name {
                        self.deps
                            .entry(var_name)
                            .or_insert_with(HashSet::new)
                            .insert(name.clone());
                    }
                }
            }
        }

        Ok(())
    }

    fn update_expressions(&mut self) -> Result<()> {
        // Get parameters with expressions
        let expr_params: Vec<_> = self
            .params
            .keys()
            .filter(|name| self.params.get(*name).unwrap().expr().is_some())
            .cloned()
            .collect();

        if expr_params.is_empty() {
            return Ok(());
        }

        // For each parameter with an expression, evaluate it
        for name in expr_params {
            let expr_str = self.params.get(&name).unwrap().expr().unwrap().to_string();
            let expr = Expression::parse(&expr_str)?;
            let value = expr.evaluate(self)?;

            let param = self.params.get_mut(&name).unwrap();
            param.set_value(value)?;
        }

        Ok(())
    }

    fn add_constraint(
        &mut self,
        lhs: &str,
        constraint_type: ConstraintType,
        rhs: &str,
    ) -> Result<()> {
        self.constraints.add_constraint(lhs, constraint_type, rhs)
    }

    fn check_constraints(&self) -> Result<bool> {
        self.constraints.all_satisfied(self)
    }

    fn violated_constraints(&self) -> Result<Vec<&Constraint>> {
        self.constraints.violated(self)
    }
}

impl EvaluationContext for Parameters {
    fn get_variable(&self, name: &str) -> Result<f64> {
        self.params
            .get(name)
            .map(|p| p.value())
            .ok_or_else(|| Error::ExpressionError(format!("Variable {} not found", name)))
    }

    fn has_variable(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }
}

// Test functions
fn test_constraint_creation() {
    // Create a constraint
    let c = Constraint::new("x", ConstraintType::LessThan, "y").unwrap();
    assert_eq!(c.to_string(), "x < y");

    // Check variables
    let vars = c.variables();
    assert_eq!(vars.len(), 2);
    assert!(vars.contains(&"x".to_string()));
    assert!(vars.contains(&"y".to_string()));

    println!("Constraint creation test passed!");
}

fn test_constraint_satisfaction() {
    // Create parameters
    let mut params = Parameters::new();
    params.add_param("x", 5.0).unwrap();
    params.add_param("y", 10.0).unwrap();

    // Create a constraint
    let c = Constraint::new("x", ConstraintType::LessThan, "y").unwrap();

    // Test satisfaction
    assert!(c.is_satisfied(&params).unwrap());

    // Change x to make constraint fail
    params.get_mut("x").unwrap().set_value(15.0).unwrap();
    assert!(!c.is_satisfied(&params).unwrap());

    println!("Constraint satisfaction test passed!");
}

fn test_parameters_with_constraints() {
    // Create parameters
    let mut params = Parameters::new();
    params.add_param("x", 5.0).unwrap();
    params.add_param("y", 10.0).unwrap();

    // Add constraints
    params
        .add_constraint("x", ConstraintType::LessThan, "y")
        .unwrap();

    // Check constraints
    assert!(params.check_constraints().unwrap());

    // Violate constraint
    params.get_mut("x").unwrap().set_value(15.0).unwrap();
    assert!(!params.check_constraints().unwrap());

    // Check violated constraints
    let violated = params.violated_constraints().unwrap();
    assert_eq!(violated.len(), 1);
    assert_eq!(violated[0].to_string(), "x < y");

    println!("Parameters with constraints test passed!");
}

fn test_parameters_with_expressions() {
    // Create parameters
    let mut params = Parameters::new();
    params.add_param("x", 2.0).unwrap();
    params.add_param("y", 3.0).unwrap();

    // Add parameter with expression
    params.add_param_with_expr("sum", 0.0, "x + y").unwrap();

    // Update expressions
    params.update_expressions().unwrap();

    // Check expression value
    assert_eq!(params.get("sum").unwrap().value(), 5.0);

    // Change base parameters
    params.get_mut("x").unwrap().set_value(4.0).unwrap();
    params.get_mut("y").unwrap().set_value(6.0).unwrap();

    // Update expressions
    params.update_expressions().unwrap();

    // Check updated expression value
    assert_eq!(params.get("sum").unwrap().value(), 10.0);

    println!("Parameters with expressions test passed!");
}

fn test_constraints_with_expressions() {
    // Create parameters
    let mut params = Parameters::new();
    params.add_param("x", 2.0).unwrap();
    params.add_param("y", 3.0).unwrap();

    // Add parameter with expression
    params.add_param_with_expr("sum", 0.0, "x + y").unwrap();

    // Update expressions
    params.update_expressions().unwrap();

    // Add constraint
    params
        .add_constraint("sum", ConstraintType::LessThan, "10")
        .unwrap();

    // Check constraints
    assert!(params.check_constraints().unwrap());

    // Change base parameters to violate constraint
    params.get_mut("x").unwrap().set_value(5.0).unwrap();
    params.get_mut("y").unwrap().set_value(6.0).unwrap();

    // Update expressions
    params.update_expressions().unwrap();

    // Check sum value
    assert_eq!(params.get("sum").unwrap().value(), 11.0);

    // Check constraints
    assert!(!params.check_constraints().unwrap());

    println!("Constraints with expressions test passed!");
}

fn main() {
    test_constraint_creation();
    test_constraint_satisfaction();
    test_parameters_with_constraints();
    test_parameters_with_expressions();
    test_constraints_with_expressions();

    println!("All tests passed!");
}
