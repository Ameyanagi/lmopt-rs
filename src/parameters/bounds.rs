//! Parameter bounds implementation
//!
//! This module provides functionality for defining and managing parameter bounds
//! similar to lmfit-py. It implements the Minuit-style parameter transformation
//! that handles bounds constraints during optimization.

use std::f64::{INFINITY, NEG_INFINITY};
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Errors that can occur when working with parameter bounds
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BoundsError {
    #[error("Invalid bounds: min ({min}) must be less than max ({max})")]
    InvalidBounds { min: f64, max: f64 },
    
    #[error("Parameter value {value} is outside bounds: [{min}, {max}]")]
    ValueOutsideBounds { value: f64, min: f64, max: f64 },
    
    #[error("Infinite parameter value is not allowed")]
    InfiniteValue,
}

/// Represents the bounds constraints on a parameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// Minimum allowed value for the parameter
    pub min: f64,
    
    /// Maximum allowed value for the parameter
    pub max: f64,
}

impl Serialize for Bounds {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        
        let mut state = serializer.serialize_struct("Bounds", 2)?;
        
        // Handle infinity values specially
        if self.min.is_infinite() && self.min.is_sign_negative() {
            state.serialize_field("min", &serde_json::Value::Null)?;
        } else {
            state.serialize_field("min", &self.min)?;
        }
        
        if self.max.is_infinite() && self.max.is_sign_positive() {
            state.serialize_field("max", &serde_json::Value::Null)?;
        } else {
            state.serialize_field("max", &self.max)?;
        }
        
        state.end()
    }
}

impl<'de> Deserialize<'de> for Bounds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct BoundsHelper {
            #[serde(default)]
            min: Option<f64>,
            
            #[serde(default)]
            max: Option<f64>,
        }
        
        let helper = BoundsHelper::deserialize(deserializer)?;
        
        let min = helper.min.unwrap_or(NEG_INFINITY);
        let max = helper.max.unwrap_or(INFINITY);
        
        Ok(Bounds { min, max })
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min: NEG_INFINITY,
            max: INFINITY,
        }
    }
}

impl Bounds {
    /// Create a new bounds constraints with min and max values
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum allowed value for the parameter
    /// * `max` - Maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// A new `Bounds` object if min <= max, or an error otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use lmopt_rs::parameters::bounds::Bounds;
    ///
    /// let bounds = Bounds::new(0.0, 10.0).unwrap();
    /// assert_eq!(bounds.min, 0.0);
    /// assert_eq!(bounds.max, 10.0);
    /// ```
    pub fn new(min: f64, max: f64) -> Result<Self, BoundsError> {
        if min > max {
            return Err(BoundsError::InvalidBounds { min, max });
        }
        
        Ok(Self { min, max })
    }
    
    /// Create an unbounded constraint (negative infinity to positive infinity)
    ///
    /// # Returns
    ///
    /// A new `Bounds` object with min = -∞ and max = ∞
    ///
    /// # Examples
    ///
    /// ```
    /// use lmopt_rs::parameters::bounds::Bounds;
    /// use std::f64::{INFINITY, NEG_INFINITY};
    ///
    /// let bounds = Bounds::unbounded();
    /// assert_eq!(bounds.min, NEG_INFINITY);
    /// assert_eq!(bounds.max, INFINITY);
    /// ```
    pub fn unbounded() -> Self {
        Self::default()
    }
    
    /// Create a bounds constraint with only a minimum value
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// A new `Bounds` object with the specified min and max = ∞
    pub fn min_only(min: f64) -> Self {
        Self {
            min,
            max: INFINITY,
        }
    }
    
    /// Create a bounds constraint with only a maximum value
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum allowed value for the parameter
    ///
    /// # Returns
    ///
    /// A new `Bounds` object with min = -∞ and the specified max
    pub fn max_only(max: f64) -> Self {
        Self {
            min: NEG_INFINITY,
            max,
        }
    }
    
    /// Check if a value is within the bounds
    ///
    /// # Arguments
    ///
    /// * `value` - Value to check
    ///
    /// # Returns
    ///
    /// `true` if the value is within the bounds, `false` otherwise
    pub fn is_within_bounds(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }
    
    /// Check if the bounds are finite (both min and max are finite)
    ///
    /// # Returns
    ///
    /// `true` if both min and max are finite, `false` otherwise
    pub fn is_finite(&self) -> bool {
        self.min.is_finite() && self.max.is_finite()
    }
    
    /// Check if the parameter is bounded from below
    ///
    /// # Returns
    ///
    /// `true` if min is finite, `false` otherwise
    pub fn has_lower_bound(&self) -> bool {
        self.min.is_finite()
    }
    
    /// Check if the parameter is bounded from above
    ///
    /// # Returns
    ///
    /// `true` if max is finite, `false` otherwise
    pub fn has_upper_bound(&self) -> bool {
        self.max.is_finite()
    }
    
    /// Clamp a value to be within the bounds
    ///
    /// # Arguments
    ///
    /// * `value` - Value to clamp
    ///
    /// # Returns
    ///
    /// The value clamped to be within the bounds
    pub fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }
}

/// Implements the Minuit-style parameter transformations for handling bounds constraints
///
/// This allows the optimizer to work with unbounded parameters internally, while the
/// external values are constrained to be within the specified bounds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundsTransform {
    bounds: Bounds,
}

impl BoundsTransform {
    /// Create a new bounds transform
    ///
    /// # Arguments
    ///
    /// * `bounds` - The bounds constraints to apply
    ///
    /// # Returns
    ///
    /// A new `BoundsTransform` object
    pub fn new(bounds: Bounds) -> Self {
        Self { bounds }
    }
    
    /// Transform an internal parameter value to an external value
    ///
    /// # Arguments
    ///
    /// * `internal_value` - The internal parameter value
    ///
    /// # Returns
    ///
    /// The corresponding external value, constrained to be within bounds
    pub fn to_external(&self, internal_value: f64) -> f64 {
        // No bound transform needed if parameter is unbounded
        if !self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            return internal_value;
        }
        
        // Only lower bound
        if self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            return self.bounds.min - 1.0 + (internal_value * internal_value + 1.0).sqrt();
        }
        
        // Only upper bound
        if !self.bounds.has_lower_bound() && self.bounds.has_upper_bound() {
            return self.bounds.max + 1.0 - (internal_value * internal_value + 1.0).sqrt();
        }
        
        // Both bounds
        let bound_range = self.bounds.max - self.bounds.min;
        self.bounds.min + (internal_value.sin() + 1.0) * bound_range / 2.0
    }
    
    /// Transform an external parameter value to an internal value
    ///
    /// # Arguments
    ///
    /// * `external_value` - The external parameter value
    ///
    /// # Returns
    ///
    /// The corresponding internal value, or an error if the external value is outside bounds
    pub fn to_internal(&self, external_value: f64) -> Result<f64, BoundsError> {
        // Check if value is finite
        if !external_value.is_finite() {
            return Err(BoundsError::InfiniteValue);
        }
        
        // Check if value is within bounds
        if !self.bounds.is_within_bounds(external_value) {
            return Err(BoundsError::ValueOutsideBounds {
                value: external_value,
                min: self.bounds.min,
                max: self.bounds.max,
            });
        }
        
        // No bound transform needed if parameter is unbounded
        if !self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            return Ok(external_value);
        }
        
        // Only lower bound
        if self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            return Ok(((external_value - self.bounds.min + 1.0).powi(2) - 1.0).sqrt());
        }
        
        // Only upper bound
        if !self.bounds.has_lower_bound() && self.bounds.has_upper_bound() {
            return Ok(((self.bounds.max - external_value + 1.0).powi(2) - 1.0).sqrt());
        }
        
        // Both bounds
        let bound_range = self.bounds.max - self.bounds.min;
        let scaled = 2.0 * (external_value - self.bounds.min) / bound_range - 1.0;
        
        // Ensure scaled is in [-1, 1] for asin
        let scaled = scaled.clamp(-1.0, 1.0);
        Ok(scaled.asin())
    }
    
    /// Scale the gradient of the objective function for a parameter
    ///
    /// # Arguments
    ///
    /// * `external_value` - The external parameter value
    /// * `gradient` - The gradient with respect to the external parameter
    ///
    /// # Returns
    ///
    /// The scaled gradient with respect to the internal parameter
    pub fn scale_gradient(&self, external_value: f64, gradient: f64) -> Result<f64, BoundsError> {
        // Check if value is within bounds
        if !self.bounds.is_within_bounds(external_value) {
            return Err(BoundsError::ValueOutsideBounds {
                value: external_value,
                min: self.bounds.min,
                max: self.bounds.max,
            });
        }
        
        // No bound transform needed if parameter is unbounded
        if !self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            return Ok(gradient);
        }
        
        // Only lower bound
        if self.bounds.has_lower_bound() && !self.bounds.has_upper_bound() {
            let internal = self.to_internal(external_value)?;
            return Ok(gradient * internal / (internal.powi(2) + 1.0).sqrt());
        }
        
        // Only upper bound
        if !self.bounds.has_lower_bound() && self.bounds.has_upper_bound() {
            let internal = self.to_internal(external_value)?;
            return Ok(-gradient * internal / (internal.powi(2) + 1.0).sqrt());
        }
        
        // Both bounds
        let internal = self.to_internal(external_value)?;
        let bound_range = self.bounds.max - self.bounds.min;
        Ok(gradient * bound_range * internal.cos() / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bounds_creation() {
        // Valid bounds
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        assert_eq!(bounds.min, 0.0);
        assert_eq!(bounds.max, 10.0);
        
        // Invalid bounds (min > max)
        let result = Bounds::new(10.0, 0.0);
        assert!(result.is_err());
        
        // Unbounded
        let bounds = Bounds::unbounded();
        assert_eq!(bounds.min, NEG_INFINITY);
        assert_eq!(bounds.max, INFINITY);
        
        // Lower bound only
        let bounds = Bounds::min_only(5.0);
        assert_eq!(bounds.min, 5.0);
        assert_eq!(bounds.max, INFINITY);
        
        // Upper bound only
        let bounds = Bounds::max_only(15.0);
        assert_eq!(bounds.min, NEG_INFINITY);
        assert_eq!(bounds.max, 15.0);
    }
    
    #[test]
    fn test_is_within_bounds() {
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        
        assert!(bounds.is_within_bounds(0.0));
        assert!(bounds.is_within_bounds(5.0));
        assert!(bounds.is_within_bounds(10.0));
        
        assert!(!bounds.is_within_bounds(-1.0));
        assert!(!bounds.is_within_bounds(11.0));
    }
    
    #[test]
    fn test_bounds_checks() {
        let unbounded = Bounds::unbounded();
        assert!(!unbounded.is_finite());
        assert!(!unbounded.has_lower_bound());
        assert!(!unbounded.has_upper_bound());
        
        let lower_bounded = Bounds::min_only(0.0);
        assert!(!lower_bounded.is_finite());
        assert!(lower_bounded.has_lower_bound());
        assert!(!lower_bounded.has_upper_bound());
        
        let upper_bounded = Bounds::max_only(10.0);
        assert!(!upper_bounded.is_finite());
        assert!(!upper_bounded.has_lower_bound());
        assert!(upper_bounded.has_upper_bound());
        
        let bounded = Bounds::new(0.0, 10.0).unwrap();
        assert!(bounded.is_finite());
        assert!(bounded.has_lower_bound());
        assert!(bounded.has_upper_bound());
    }
    
    #[test]
    fn test_clamp() {
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        
        assert_eq!(bounds.clamp(-5.0), 0.0);
        assert_eq!(bounds.clamp(5.0), 5.0);
        assert_eq!(bounds.clamp(15.0), 10.0);
    }
    
    #[test]
    fn test_bounds_transform_unbounded() {
        let bounds = Bounds::unbounded();
        let transform = BoundsTransform::new(bounds);
        
        // For unbounded parameters, internal and external values should be the same
        let test_values = [-10.0, -1.0, 0.0, 1.0, 10.0];
        
        for &value in &test_values {
            assert_eq!(transform.to_external(value), value);
            assert_eq!(transform.to_internal(value).unwrap(), value);
        }
    }
    
    #[test]
    fn test_bounds_transform_lower_bound() {
        let bounds = Bounds::min_only(5.0);
        let transform = BoundsTransform::new(bounds);
        
        // Test converting from internal to external values
        let internal_values = [1.0, 5.0, 10.0];
        for &internal in &internal_values {
            let external = transform.to_external(internal);
            assert!(external >= bounds.min);
            
            // Round-trip test with looser tolerance
            let internal_round_trip = transform.to_internal(external).unwrap();
            assert!((internal - internal_round_trip).abs() < 1e-8,
                    "Round-trip difference: {}", (internal - internal_round_trip).abs());
        }
    }
    
    #[test]
    fn test_bounds_transform_upper_bound() {
        let bounds = Bounds::max_only(5.0);
        let transform = BoundsTransform::new(bounds);
        
        // Test converting from internal to external values
        let internal_values = [1.0, 5.0, 10.0];
        for &internal in &internal_values {
            let external = transform.to_external(internal);
            assert!(external <= bounds.max);
            
            // Round-trip test with looser tolerance
            let internal_round_trip = transform.to_internal(external).unwrap();
            assert!((internal - internal_round_trip).abs() < 1e-8,
                    "Round-trip difference: {}", (internal - internal_round_trip).abs());
        }
    }
    
    #[test]
    fn test_bounds_transform_both_bounds() {
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        let transform = BoundsTransform::new(bounds);
        
        // Test converting from internal to external values
        let internal_values = [0.0, 0.5, 1.0];
        for &internal in &internal_values {
            let external = transform.to_external(internal);
            assert!(external >= bounds.min);
            assert!(external <= bounds.max);
            
            // Round-trip test with looser tolerance
            let internal_round_trip = transform.to_internal(external).unwrap();
            assert!((internal - internal_round_trip).abs() < 1e-8,
                    "Round-trip difference: {}", (internal - internal_round_trip).abs());
        }
    }
    
    #[test]
    fn test_bounds_transform_errors() {
        let bounds = Bounds::new(0.0, 10.0).unwrap();
        let transform = BoundsTransform::new(bounds);
        
        // Value out of bounds
        assert!(transform.to_internal(-1.0).is_err());
        assert!(transform.to_internal(11.0).is_err());
        
        // Infinite value
        assert!(transform.to_internal(INFINITY).is_err());
        assert!(transform.to_internal(NEG_INFINITY).is_err());
    }
}