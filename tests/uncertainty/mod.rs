// Tests for uncertainty calculations

#[cfg(test)]
mod tests {
    use lmopt_rs::parameters::{Parameter, Parameters};
    use lmopt_rs::{
        covariance_matrix, standard_errors, uncertainty_analysis, ConfidenceInterval,
        UncertaintyCalculator, UncertaintyResult,
    };
    use ndarray::{arr1, arr2, Array2};

    /// Create a mock jacobian for a linear fit: y = mx + b
    /// where m is parameter 0 and b is parameter 1
    fn create_mock_jacobian() -> Array2<f64> {
        // For points (1,1), (2,2), (3,2), (4,3)
        // J = [dmodel/dm, dmodel/db] = [x, 1]
        arr2(&[
            [1.0, 1.0], // x=1
            [2.0, 1.0], // x=2
            [3.0, 1.0], // x=3
            [4.0, 1.0], // x=4
        ])
    }

    /// Create mock parameters for a linear fit
    fn create_mock_parameters() -> Parameters {
        let mut params = Parameters::new();
        let mut m = Parameter::new("m", 0.7);
        let mut b = Parameter::new("b", 0.3);

        // Set parameters to vary
        m.set_vary(true).unwrap();
        b.set_vary(true).unwrap();

        params.add(m).unwrap(); // Slope
        params.add(b).unwrap(); // Intercept
        params
    }

    #[test]
    #[cfg(feature = "matrix")]
    fn test_covariance_and_standard_errors() {
        let jacobian = create_mock_jacobian();
        let params = create_mock_parameters();

        // For y = mx + b fit to (1,1), (2,2), (3,2), (4,3)
        // residuals = [0, 0, 0.3, 0.2]
        // chisqr = 0.13
        let chisqr = 0.13;
        let ndata = 4;
        let nvarys = 2;

        // Calculate covariance matrix
        let covar = covariance_matrix(&jacobian, chisqr, ndata, nvarys).unwrap();

        // Check covariance matrix shape
        assert_eq!(covar.shape(), &[2, 2]);

        // Calculate standard errors
        let errors = standard_errors(&covar, &params);

        // Check that we have standard errors for both parameters
        assert!(errors.contains_key("m"));
        assert!(errors.contains_key("b"));

        // Standard errors should be positive
        assert!(errors["m"] > 0.0);
        assert!(errors["b"] > 0.0);

        // Create uncertainty calculator
        let calc = UncertaintyCalculator::new(ndata, nvarys, chisqr);

        // Check properties
        assert_eq!(calc.nfree, 2); // ndata - nvarys = 4 - 2 = 2
        assert_eq!(calc.chisqr, chisqr);
        assert!((calc.redchi - 0.065).abs() < 1e-10); // 0.13 / 2 = 0.065

        // Test correlation matrix
        let correl = calc.calculate_correlation(&covar);

        // Correlation matrix should have diagonal elements = 1.0
        assert_eq!(correl.shape(), &[2, 2]);
        assert_eq!(correl[[0, 0]], 1.0);
        assert_eq!(correl[[1, 1]], 1.0);

        // Off-diagonal elements should be between -1 and 1
        assert!(correl[[0, 1]].abs() <= 1.0);
        assert!(correl[[1, 0]].abs() <= 1.0);

        // Correlation matrix should be symmetric
        assert_eq!(correl[[0, 1]], correl[[1, 0]]);
    }

    #[test]
    #[cfg(feature = "matrix")]
    fn test_full_uncertainty_analysis() {
        let jacobian = create_mock_jacobian();
        let params = create_mock_parameters();

        let chisqr = 0.13;
        let ndata = 4;
        let sigmas = &[1.0, 2.0, 3.0]; // 1, 2, and 3 sigma confidence intervals

        // Perform full uncertainty analysis
        let result = uncertainty_analysis(&jacobian, &params, chisqr, ndata, sigmas).unwrap();

        // Check that we have all the expected components
        assert_eq!(result.covariance.shape(), &[2, 2]);
        assert_eq!(result.correlation.shape(), &[2, 2]);
        assert!(result.standard_errors.contains_key("m"));
        assert!(result.standard_errors.contains_key("b"));

        // Confidence intervals not fully implemented yet, but should at least be an empty map
        assert!(result.confidence_intervals.is_empty());
    }

    #[test]
    fn test_f_test() {
        let calc = UncertaintyCalculator::new(10, 2, 100.0);

        // Model 1: chisqr = 100, nfree = 8
        // Model 2: chisqr = 120, nfree = 7 (one parameter fixed)
        // dchi = 120/100 - 1 = 0.2
        // f = 0.2 * 7 / (8-7) = 0.2 * 7 = 1.4
        let f_stat = calc.f_test(100.0, 8, 120.0, 7);
        assert!((f_stat - 1.4).abs() < 1e-10);

        // If model 2 has more free parameters than model 1, return 1.0
        let f_invalid = calc.f_test(100.0, 7, 120.0, 8);
        assert_eq!(f_invalid, 1.0);
    }
}
