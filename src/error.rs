use thiserror::Error;

/// Error types for the lmopt-rs library.
#[derive(Error, Debug)]
pub enum LmOptError {
    /// Error indicating a mismatch in matrix dimensions.
    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Error during matrix conversion operations.
    #[error("Matrix conversion error: {0}")]
    ConversionError(String),

    /// Error indicating a singular matrix was encountered.
    #[error("Singular matrix encountered")]
    SingularMatrix,

    /// Error indicating the algorithm failed to converge.
    #[error("Algorithm failed to converge: {0}")]
    ConvergenceFailure(String),

    /// Error indicating optimization failed.
    #[error("Optimization failed: {0}")]
    OptimizationFailure(String),

    /// Error for invalid parameter values.
    #[error("Invalid parameter value: {0}")]
    InvalidParameter(String),

    /// Error for parameter-related problems.
    #[error("Parameter error: {0}")]
    ParameterError(String),

    /// Error for boundary constraint violations.
    #[error("Bounds error: {0}")]
    BoundsError(String),

    /// Error during function evaluation.
    #[error("Function evaluation error: {0}")]
    FunctionEvaluation(String),

    /// Error during computational operations.
    #[error("Computation error: {0}")]
    InvalidComputation(String),

    /// Error during computational processing.
    #[error("Computation error: {0}")]
    ComputationError(String),

    /// Not implemented functionality.
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Invalid input data.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid state in the algorithm or data structure.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Parameter not found.
    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    /// Linear algebra error.
    #[error("Linear algebra error: {0}")]
    LinearAlgebraError(String),

    /// I/O error wrapper.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Generic error for cases that don't fit the other categories.
    #[error("Error: {0}")]
    Other(String),
}

// Add From implementation for ParameterError
impl From<crate::parameters::parameter::ParameterError> for LmOptError {
    fn from(err: crate::parameters::parameter::ParameterError) -> Self {
        LmOptError::ParameterError(format!("{}", err))
    }
}

/// Result type alias for lmopt-rs operations.
pub type Result<T> = std::result::Result<T, LmOptError>;

/// Extensions for converting from other error types.
impl From<String> for LmOptError {
    fn from(s: String) -> Self {
        LmOptError::Other(s)
    }
}

impl From<&str> for LmOptError {
    fn from(s: &str) -> Self {
        LmOptError::Other(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LmOptError::DimensionMismatch("expected 3x3, got 2x2".to_string());
        assert!(format!("{}", err).contains("expected 3x3, got 2x2"));

        let err = LmOptError::ConvergenceFailure("exceeded max iterations".to_string());
        assert!(format!("{}", err).contains("exceeded max iterations"));
    }

    #[test]
    fn test_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: LmOptError = io_err.into();

        match err {
            LmOptError::IoError(_) => (),
            _ => panic!("Expected IoError variant"),
        }

        let str_err: LmOptError = "test error".into();
        match str_err {
            LmOptError::Other(s) => assert_eq!(s, "test error"),
            _ => panic!("Expected Other variant"),
        }
    }
}
