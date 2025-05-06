//! Parameter definition and implementation
//!
//! This module provides the Parameter struct, which is the fundamental building
//! block of the parameter system. Parameters can be varied during optimization,
//! can have bounds constraints, and can be linked through expressions.

use crate::parameters::bounds::{Bounds, BoundsTransform, BoundsError};
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Errors that can occur when working with parameters
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParameterError {
    #[error("Parameter '{name}' cannot be both fixed and have an expression")]
    FixedWithExpression { name: String },
    
    #[error("Parameter '{name}' cannot have both an expression and be varied")]
    ExpressionAndVary { name: String },
    
    #[error("Bounds error: {0}")]
    BoundsError(#[from] BoundsError),
    
    #[error("Cannot evaluate expression for parameter '{name}': {message}")]
    ExpressionEvaluation { name: String, message: String },
    
    #[error("Parameter '{name}' not found")]
    ParameterNotFound { name: String },
    
    #[error("Circular dependency in expression for parameter '{name}'")]
    CircularDependency { name: String },
}

/// A parameter for optimization problems
///
/// Parameters can be varied during optimization, can have bounds constraints,
/// and can be linked through expressions. This is similar to lmfit-py's Parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Name of the parameter
    pub name: String,
    
    /// Current value of the parameter
    value: f64,
    
    /// Initial value when created (for reset operations)
    init_value: f64,
    
    /// Whether this parameter can be varied during optimization
    pub vary: bool,
    
    /// Minimum and maximum bounds for the parameter value
    bounds: Bounds,
    
    /// The expression used to compute this parameter (if any)
    pub expr: Option<String>,
    
    /// Standard error of the parameter (set after fitting)
    pub stderr: Option<f64>,
    
    /// Step size for brute-force methods
    pub brute_step: Option<f64>,
    
    /// User data associated with this parameter
    pub user_data: Option<String>,
}

impl Parameter {
    /// Create a new parameter with the given name and value
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the parameter
    /// * `value` - Initial value of the parameter
    ///
    /// # Returns
    ///
    /// A new parameter with the given name and value. The parameter will be varied
    /// during optimization, and will have no bounds constraints.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmopt_rs::parameters::parameter::Parameter;
    ///
    /// let param = Parameter::new("amplitude", 10.0);
    /// assert_eq!(param.name(), "amplitude");
    /// assert_eq!(param.value(), 10.0);
    /// assert!(param.vary());
    /// ```
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            init_value: value,
            vary: true,
            bounds: Bounds::default(),
            expr: None,
            stderr: None,
            brute_step: None,
            user_data: None,
        }
    }
    
    /// Create a new parameter with the given name, value, and bounds
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the parameter
    /// * `value` - Initial value of the parameter
    /// * `min` - Minimum allowed value for the parameter
    /// * `max` - Maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// A new parameter with the given name, value, and bounds. The parameter will be varied
    /// during optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmopt_rs::parameters::parameter::Parameter;
    ///
    /// let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
    /// assert_eq!(param.name(), "amplitude");
    /// assert_eq!(param.value(), 10.0);
    /// assert!(param.vary());
    /// assert_eq!(param.min(), 0.0);
    /// assert_eq!(param.max(), 20.0);
    /// ```
    pub fn with_bounds(name: &str, value: f64, min: f64, max: f64) -> Result<Self, ParameterError> {
        let bounds = Bounds::new(min, max)?;
        
        // Clamp the value to be within bounds
        let value = bounds.clamp(value);
        
        Ok(Self {
            name: name.to_string(),
            value,
            init_value: value,
            vary: true,
            bounds,
            expr: None,
            stderr: None,
            brute_step: None,
            user_data: None,
        })
    }
    
    /// Create a new parameter with the given name, value, and expression
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the parameter
    /// * `value` - Initial value of the parameter
    /// * `expr` - Expression to compute the parameter value
    ///
    /// # Returns
    ///
    /// A new parameter with the given name, value, and expression. The parameter will not be varied
    /// during optimization, since its value is determined by the expression.
    ///
    /// # Examples
    ///
    /// ```
    /// use lmopt_rs::parameters::parameter::Parameter;
    ///
    /// let param = Parameter::with_expr("half_amplitude", 5.0, "amplitude / 2").unwrap();
    /// assert_eq!(param.name(), "half_amplitude");
    /// assert_eq!(param.value(), 5.0);
    /// assert!(!param.vary());
    /// assert_eq!(param.expr().unwrap(), "amplitude / 2");
    /// ```
    pub fn with_expr(name: &str, value: f64, expr: &str) -> Result<Self, ParameterError> {
        let expr_str = expr.to_string();
        
        Ok(Self {
            name: name.to_string(),
            value,
            init_value: value,
            vary: false,
            bounds: Bounds::default(),
            expr: Some(expr_str),
            stderr: None,
            brute_step: None,
            user_data: None,
        })
    }
    
    /// Get the current value of the parameter
    ///
    /// # Returns
    ///
    /// The current value of the parameter
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Set the value of the parameter
    ///
    /// # Arguments
    ///
    /// * `value` - The new value for the parameter
    ///
    /// # Returns
    ///
    /// `Ok(())` if the value was set successfully, or an error if the value is outside bounds
    pub fn set_value(&mut self, value: f64) -> Result<(), ParameterError> {
        // Check if value is within bounds
        if !self.bounds.is_within_bounds(value) {
            return Err(ParameterError::BoundsError(BoundsError::ValueOutsideBounds {
                value,
                min: self.bounds.min,
                max: self.bounds.max,
            }));
        }
        
        self.value = value;
        Ok(())
    }
    
    /// Get the initial value of the parameter
    ///
    /// # Returns
    ///
    /// The initial value of the parameter when it was created
    pub fn init_value(&self) -> f64 {
        self.init_value
    }
    
    /// Reset the parameter to its initial value
    pub fn reset(&mut self) {
        // Use clamp to ensure the initial value is within current bounds
        self.value = self.bounds.clamp(self.init_value);
    }
    
    /// Get the name of the parameter
    ///
    /// # Returns
    ///
    /// The name of the parameter
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Set the name of the parameter
    ///
    /// # Arguments
    ///
    /// * `name` - The new name for the parameter
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
    
    /// Check if the parameter is varied during optimization
    ///
    /// # Returns
    ///
    /// `true` if the parameter is varied during optimization, `false` otherwise
    pub fn vary(&self) -> bool {
        self.vary
    }
    
    /// Set whether the parameter is varied during optimization
    ///
    /// # Arguments
    ///
    /// * `vary` - Whether the parameter should be varied during optimization
    ///
    /// # Returns
    ///
    /// `Ok(())` if the value was set successfully, or an error if the parameter has an expression
    pub fn set_vary(&mut self, vary: bool) -> Result<(), ParameterError> {
        // Cannot vary a parameter with an expression
        if vary && self.expr.is_some() {
            return Err(ParameterError::ExpressionAndVary {
                name: self.name.clone(),
            });
        }
        
        self.vary = vary;
        Ok(())
    }
    
    /// Get the minimum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// The minimum allowed value for the parameter
    pub fn min(&self) -> f64 {
        self.bounds.min
    }
    
    /// Get the maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// The maximum allowed value for the parameter
    pub fn max(&self) -> f64 {
        self.bounds.max
    }
    
    /// Set the bounds for the parameter
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum allowed value for the parameter
    /// * `max` - Maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// `Ok(())` if the bounds were set successfully, or an error if min > max
    pub fn set_bounds(&mut self, min: f64, max: f64) -> Result<(), ParameterError> {
        let bounds = Bounds::new(min, max)?;
        self.bounds = bounds;
        
        // Ensure value is within bounds
        self.value = bounds.clamp(self.value);
        
        Ok(())
    }
    
    /// Set the minimum bound for the parameter
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// `Ok(())` if the bound was set successfully
    pub fn set_min(&mut self, min: f64) -> Result<(), ParameterError> {
        self.set_bounds(min, self.bounds.max)
    }
    
    /// Set the maximum bound for the parameter
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// `Ok(())` if the bound was set successfully
    pub fn set_max(&mut self, max: f64) -> Result<(), ParameterError> {
        self.set_bounds(self.bounds.min, max)
    }
    
    /// Get the expression used to compute this parameter (if any)
    ///
    /// # Returns
    ///
    /// The expression used to compute this parameter, or `None` if the parameter has no expression
    pub fn expr(&self) -> Option<&str> {
        self.expr.as_deref()
    }
    
    /// Set the expression used to compute this parameter
    ///
    /// # Arguments
    ///
    /// * `expr` - Expression to compute the parameter value, or `None` to remove the expression
    ///
    /// # Returns
    ///
    /// `Ok(())` if the expression was set successfully, or an error if the parameter is fixed
    pub fn set_expr(&mut self, expr: Option<&str>) -> Result<(), ParameterError> {
        match expr {
            Some(expr_str) => {
                // Setting an expression automatically makes the parameter not vary
                self.expr = Some(expr_str.to_string());
                self.vary = false;
            }
            None => {
                self.expr = None;
                // Removing an expression doesn't automatically make the parameter vary
                // that's up to the caller
            }
        }
        
        Ok(())
    }
    
    /// Get the standard error of the parameter (if available)
    ///
    /// # Returns
    ///
    /// The standard error of the parameter, or `None` if not available
    pub fn stderr(&self) -> Option<f64> {
        self.stderr
    }
    
    /// Set the standard error of the parameter
    ///
    /// # Arguments
    ///
    /// * `stderr` - Standard error of the parameter, or `None` to remove it
    pub fn set_stderr(&mut self, stderr: Option<f64>) {
        self.stderr = stderr;
    }
    
    /// Get the brute-force step size (if available)
    ///
    /// # Returns
    ///
    /// The brute-force step size, or `None` if not available
    pub fn brute_step(&self) -> Option<f64> {
        self.brute_step
    }
    
    /// Set the brute-force step size
    ///
    /// # Arguments
    ///
    /// * `brute_step` - Step size for brute-force methods, or `None` to remove it
    pub fn set_brute_step(&mut self, brute_step: Option<f64>) {
        self.brute_step = brute_step;
    }
    
    /// Get the user data associated with this parameter (if any)
    ///
    /// # Returns
    ///
    /// The user data associated with this parameter, or `None` if not available
    pub fn user_data(&self) -> Option<&str> {
        self.user_data.as_deref()
    }
    
    /// Set the user data associated with this parameter
    ///
    /// # Arguments
    ///
    /// * `user_data` - User data to associate with this parameter, or `None` to remove it
    pub fn set_user_data(&mut self, user_data: Option<&str>) {
        self.user_data = user_data.map(|s| s.to_string());
    }
    
    /// Get the bounds of the parameter
    ///
    /// # Returns
    ///
    /// The bounds of the parameter
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }
    
    /// Create a bounds transform for this parameter
    ///
    /// # Returns
    ///
    /// A bounds transform for this parameter
    pub fn bounds_transform(&self) -> BoundsTransform {
        BoundsTransform::new(self.bounds)
    }
    
    /// Convert the parameter value to an internal value for the optimizer
    ///
    /// # Returns
    ///
    /// The internal value for the optimizer, or an error if the conversion fails
    pub fn to_internal(&self) -> Result<f64, ParameterError> {
        let transform = self.bounds_transform();
        transform.to_internal(self.value).map_err(ParameterError::from)
    }
    
    /// Convert an internal value from the optimizer to a parameter value
    ///
    /// # Arguments
    ///
    /// * `internal_value` - The internal value from the optimizer
    ///
    /// # Returns
    ///
    /// The corresponding parameter value
    pub fn from_internal(&self, internal_value: f64) -> f64 {
        let transform = self.bounds_transform();
        transform.to_external(internal_value)
    }
    
    /// Scale the gradient of the objective function for this parameter
    ///
    /// # Arguments
    ///
    /// * `gradient` - The gradient with respect to the external parameter
    ///
    /// # Returns
    ///
    /// The scaled gradient with respect to the internal parameter
    pub fn scale_gradient(&self, gradient: f64) -> Result<f64, ParameterError> {
        let transform = self.bounds_transform();
        transform.scale_gradient(self.value, gradient).map_err(ParameterError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::{INFINITY, NEG_INFINITY};
    
    #[test]
    fn test_parameter_creation() {
        // Basic parameter
        let param = Parameter::new("amplitude", 10.0);
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);
        assert_eq!(param.init_value(), 10.0);
        assert!(param.vary());
        assert_eq!(param.min(), NEG_INFINITY);
        assert_eq!(param.max(), INFINITY);
        assert!(param.expr().is_none());
        
        // Parameter with bounds
        let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        assert_eq!(param.name(), "amplitude");
        assert_eq!(param.value(), 10.0);
        assert!(param.vary());
        assert_eq!(param.min(), 0.0);
        assert_eq!(param.max(), 20.0);
        
        // Parameter with expression
        let param = Parameter::with_expr("half_amplitude", 5.0, "amplitude / 2").unwrap();
        assert_eq!(param.name(), "half_amplitude");
        assert_eq!(param.value(), 5.0);
        assert!(!param.vary());
        assert_eq!(param.expr().unwrap(), "amplitude / 2");
    }
    
    #[test]
    fn test_parameter_value() {
        // Test setting and getting values
        let mut param = Parameter::new("amplitude", 10.0);
        assert_eq!(param.value(), 10.0);
        
        param.set_value(15.0).unwrap();
        assert_eq!(param.value(), 15.0);
        
        // Test value clamping with bounds
        let mut param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        
        // Valid value
        param.set_value(15.0).unwrap();
        assert_eq!(param.value(), 15.0);
        
        // Value outside bounds should return an error
        assert!(param.set_value(25.0).is_err());
        assert_eq!(param.value(), 15.0);
        
        assert!(param.set_value(-5.0).is_err());
        assert_eq!(param.value(), 15.0);
    }
    
    #[test]
    fn test_parameter_reset() {
        let mut param = Parameter::new("amplitude", 10.0);
        param.set_value(15.0).unwrap();
        assert_eq!(param.value(), 15.0);
        
        param.reset();
        assert_eq!(param.value(), 10.0);
        
        // Test reset with bounds
        let mut param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        param.set_value(15.0).unwrap();
        assert_eq!(param.value(), 15.0);
        
        param.reset();
        assert_eq!(param.value(), 10.0);
        
        // Test reset with changed bounds
        let mut param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        param.set_value(15.0).unwrap();
        param.set_bounds(5.0, 15.0).unwrap();
        
        param.reset();
        assert_eq!(param.value(), 10.0);
        
        // If initial value is outside new bounds, it should be clamped
        let mut param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        param.set_value(15.0).unwrap();
        param.set_bounds(12.0, 18.0).unwrap();
        
        param.reset();
        assert_eq!(param.value(), 12.0);
    }
    
    #[test]
    fn test_parameter_vary() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert!(param.vary());
        
        param.set_vary(false).unwrap();
        assert!(!param.vary());
        
        param.set_vary(true).unwrap();
        assert!(param.vary());
        
        // Test vary with expression
        let mut param = Parameter::with_expr("half_amplitude", 5.0, "amplitude / 2").unwrap();
        assert!(!param.vary());
        
        // Cannot vary a parameter with an expression
        assert!(param.set_vary(true).is_err());
        assert!(!param.vary());
    }
    
    #[test]
    fn test_parameter_bounds() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert_eq!(param.min(), NEG_INFINITY);
        assert_eq!(param.max(), INFINITY);
        
        param.set_bounds(0.0, 20.0).unwrap();
        assert_eq!(param.min(), 0.0);
        assert_eq!(param.max(), 20.0);
        
        // Invalid bounds should return an error
        assert!(param.set_bounds(20.0, 0.0).is_err());
        assert_eq!(param.min(), 0.0);
        assert_eq!(param.max(), 20.0);
        
        // Setting bounds that would make current value invalid should clamp the value
        let mut param = Parameter::new("amplitude", 10.0);
        param.set_bounds(15.0, 25.0).unwrap();
        assert_eq!(param.value(), 15.0);
        
        let mut param = Parameter::new("amplitude", 10.0);
        param.set_bounds(0.0, 5.0).unwrap();
        assert_eq!(param.value(), 5.0);
    }
    
    #[test]
    fn test_parameter_expr() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert!(param.expr().is_none());
        
        param.set_expr(Some("2 * other_param")).unwrap();
        assert_eq!(param.expr().unwrap(), "2 * other_param");
        assert!(!param.vary());
        
        param.set_expr(None).unwrap();
        assert!(param.expr().is_none());
        assert!(!param.vary());
        
        // Setting vary to true and then setting an expression should work
        param.set_vary(true).unwrap();
        param.set_expr(Some("2 * other_param")).unwrap();
        assert_eq!(param.expr().unwrap(), "2 * other_param");
        assert!(!param.vary());
    }
    
    #[test]
    fn test_parameter_stderr() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert!(param.stderr().is_none());
        
        param.set_stderr(Some(0.5));
        assert_eq!(param.stderr().unwrap(), 0.5);
        
        param.set_stderr(None);
        assert!(param.stderr().is_none());
    }
    
    #[test]
    fn test_parameter_brute_step() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert!(param.brute_step().is_none());
        
        param.set_brute_step(Some(0.1));
        assert_eq!(param.brute_step().unwrap(), 0.1);
        
        param.set_brute_step(None);
        assert!(param.brute_step().is_none());
    }
    
    #[test]
    fn test_parameter_user_data() {
        let mut param = Parameter::new("amplitude", 10.0);
        assert!(param.user_data().is_none());
        
        param.set_user_data(Some("Custom data"));
        assert_eq!(param.user_data().unwrap(), "Custom data");
        
        param.set_user_data(None);
        assert!(param.user_data().is_none());
    }
    
    #[test]
    fn test_parameter_bounds_transform() {
        // Test unbounded parameter
        let param = Parameter::new("amplitude", 10.0);
        
        let internal = param.to_internal().unwrap();
        assert_eq!(internal, 10.0);
        
        let external = param.from_internal(15.0);
        assert_eq!(external, 15.0);
        
        // Test bounded parameter
        let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        
        let internal = param.to_internal().unwrap();
        let external = param.from_internal(internal);
        assert!((external - 10.0).abs() < 1e-10);
        
        // Test gradient scaling
        let param = Parameter::with_bounds("amplitude", 10.0, 0.0, 20.0).unwrap();
        
        let scaled_grad = param.scale_gradient(1.0).unwrap();
        assert!(scaled_grad.is_finite());
    }
}