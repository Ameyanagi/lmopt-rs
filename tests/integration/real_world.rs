//! Integration tests for real-world optimization problems
//!
//! These tests demonstrate practical applications of the Levenberg-Marquardt
//! algorithm for solving real-world problems.

use lmopt_rs::{
    lm::{LevenbergMarquardt, LevenbergMarquardtResult, LMConfig},
    problem::Problem,
};
use ndarray::{Array1, Array2};
use std::error::Error;

/// A rate law model for chemical kinetics
/// Fits the Arrhenius equation: k = A * exp(-Ea/RT)
struct ArrheniusModel {
    temperature: Array1<f64>, // Temperature in Kelvin
    rate_constant: Array1<f64>, // Observed rate constants
}

impl ArrheniusModel {
    fn new(temperature: Array1<f64>, rate_constant: Array1<f64>) -> Self {
        Self {
            temperature,
            rate_constant,
        }
    }
    
    /// Arrhenius equation: k = A * exp(-Ea/RT)
    /// - params[0]: ln(A) - logarithm of pre-exponential factor
    /// - params[1]: Ea - activation energy in J/mol
    fn arrhenius(temperature: f64, ln_a: f64, ea: f64) -> f64 {
        let r = 8.314; // Gas constant in J/(mol·K)
        (ln_a).exp() * (-ea / (r * temperature)).exp()
    }
}

impl Problem for ArrheniusModel {
    type Scalar = f64;
    type Parameters = Array1<f64>;
    type ResidualVector = Array1<f64>;
    type JacobianMatrix = Array2<f64>;

    fn residuals(&self, params: &Self::Parameters) -> Self::ResidualVector {
        let ln_a = params[0];
        let ea = params[1];
        
        let predicted = self.temperature.mapv(|temp| Self::arrhenius(temp, ln_a, ea));
        &predicted - &self.rate_constant
    }

    fn jacobian(&self, params: &Self::Parameters) -> Self::JacobianMatrix {
        let ln_a = params[0];
        let ea = params[1];
        let r = 8.314; // Gas constant
        let n = self.temperature.len();
        
        let mut jacobian = Array2::<f64>::zeros((n, 2));
        
        for i in 0..n {
            let temp = self.temperature[i];
            let exp_term = (-ea / (r * temp)).exp();
            let a = (ln_a).exp();
            
            // ∂f/∂ln(A) = A * exp(-Ea/RT)
            jacobian[[i, 0]] = a * exp_term;
            
            // ∂f/∂Ea = -A * exp(-Ea/RT) / (RT)
            jacobian[[i, 1]] = -a * exp_term / (r * temp);
        }
        
        jacobian
    }

    fn num_residuals(&self) -> usize {
        self.temperature.len()
    }

    fn num_parameters(&self) -> usize {
        2 // ln(A) and Ea
    }
}

/// Test fitting the Arrhenius model to synthetic data
#[test]
fn test_arrhenius_model() {
    // Create synthetic data for a chemical reaction
    // Temperatures in Kelvin
    let temperatures = Array1::from_vec(vec![
        300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0
    ]);
    
    // True parameters: ln(A) = 20, Ea = 50000 J/mol
    let true_ln_a = 20.0;
    let true_ea = 50000.0;
    
    // Generate synthetic rate constants with the Arrhenius equation
    let rate_constants = temperatures.mapv(|temp| {
        ArrheniusModel::arrhenius(temp, true_ln_a, true_ea)
    });
    
    // Create model
    let model = ArrheniusModel::new(temperatures, rate_constants);
    
    // Initial guess for parameters (intentionally off)
    let initial_params = Array1::from_vec(vec![15.0, 40000.0]);
    
    // Configure and run LM algorithm
    let config = LMConfig::default()
        .max_iterations(100)
        .parameter_tolerance(1e-6)
        .function_tolerance(1e-6);
    
    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, &initial_params);
    
    match result {
        LevenbergMarquardtResult::Success(solution) => {
            // Extract optimized parameters
            let optimized_params = solution.params;
            
            // Check that parameters are close to true values
            let ln_a_tolerance = 1e-3;
            let ea_tolerance = 1.0; // J/mol tolerance
            
            assert!((optimized_params[0] - true_ln_a).abs() < ln_a_tolerance,
                "ln(A) doesn't match: got {}, expected {}", optimized_params[0], true_ln_a);
                
            assert!((optimized_params[1] - true_ea).abs() < ea_tolerance,
                "Ea doesn't match: got {}, expected {}", optimized_params[1], true_ea);
            
            println!("Arrhenius model test passed! Parameters: ln(A) = {}, Ea = {} J/mol", 
                     optimized_params[0], optimized_params[1]);
        }
        LevenbergMarquardtResult::Error(err) => {
            panic!("LM optimization failed: {:?}", err);
        }
    }
}

/// A compartmental model for pharmacokinetics
/// Two-compartment model with intravenous bolus injection
struct PharmacokineticModel {
    time: Array1<f64>,       // Time points in hours
    concentration: Array1<f64>, // Plasma drug concentration
}

impl PharmacokineticModel {
    fn new(time: Array1<f64>, concentration: Array1<f64>) -> Self {
        Self {
            time,
            concentration,
        }
    }
    
    /// Two-compartment model: C(t) = A * exp(-alpha * t) + B * exp(-beta * t)
    /// - params[0]: A - coefficient of first exponential
    /// - params[1]: alpha - first rate constant
    /// - params[2]: B - coefficient of second exponential
    /// - params[3]: beta - second rate constant
    fn two_compartment(time: f64, a: f64, alpha: f64, b: f64, beta: f64) -> f64 {
        a * (-alpha * time).exp() + b * (-beta * time).exp()
    }
}

impl Problem for PharmacokineticModel {
    type Scalar = f64;
    type Parameters = Array1<f64>;
    type ResidualVector = Array1<f64>;
    type JacobianMatrix = Array2<f64>;

    fn residuals(&self, params: &Self::Parameters) -> Self::ResidualVector {
        let a = params[0];
        let alpha = params[1];
        let b = params[2];
        let beta = params[3];
        
        let predicted = self.time.mapv(|t| Self::two_compartment(t, a, alpha, b, beta));
        &predicted - &self.concentration
    }

    fn jacobian(&self, params: &Self::Parameters) -> Self::JacobianMatrix {
        let a = params[0];
        let alpha = params[1];
        let b = params[2];
        let beta = params[3];
        let n = self.time.len();
        
        let mut jacobian = Array2::<f64>::zeros((n, 4));
        
        for i in 0..n {
            let t = self.time[i];
            let exp_alpha = (-alpha * t).exp();
            let exp_beta = (-beta * t).exp();
            
            // ∂f/∂A = exp(-alpha * t)
            jacobian[[i, 0]] = exp_alpha;
            
            // ∂f/∂alpha = -A * t * exp(-alpha * t)
            jacobian[[i, 1]] = -a * t * exp_alpha;
            
            // ∂f/∂B = exp(-beta * t)
            jacobian[[i, 2]] = exp_beta;
            
            // ∂f/∂beta = -B * t * exp(-beta * t)
            jacobian[[i, 3]] = -b * t * exp_beta;
        }
        
        jacobian
    }

    fn num_residuals(&self) -> usize {
        self.time.len()
    }

    fn num_parameters(&self) -> usize {
        4 // A, alpha, B, beta
    }
}

/// Test fitting the pharmacokinetic model to synthetic data
#[test]
fn test_pharmacokinetic_model() {
    // Create synthetic data for drug concentration over time
    // Time points in hours
    let time_points = Array1::from_vec(vec![
        0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0
    ]);
    
    // True parameters: A = 10, alpha = 1.2, B = 5, beta = 0.2
    let true_a = 10.0;
    let true_alpha = 1.2;
    let true_b = 5.0;
    let true_beta = 0.2;
    
    // Generate synthetic concentrations with the two-compartment model
    let concentrations = time_points.mapv(|t| {
        PharmacokineticModel::two_compartment(t, true_a, true_alpha, true_b, true_beta)
    });
    
    // Create model
    let model = PharmacokineticModel::new(time_points, concentrations);
    
    // Initial guess for parameters (intentionally off)
    let initial_params = Array1::from_vec(vec![8.0, 1.0, 4.0, 0.15]);
    
    // Configure and run LM algorithm
    let config = LMConfig::default()
        .max_iterations(200)
        .parameter_tolerance(1e-6)
        .function_tolerance(1e-6);
    
    let lm = LevenbergMarquardt::new(config);
    let result = lm.minimize(&model, &initial_params);
    
    match result {
        LevenbergMarquardtResult::Success(solution) => {
            // Extract optimized parameters
            let optimized_params = solution.params;
            
            // Check that parameters are close to true values
            let tolerance = 1e-2;
            
            assert!((optimized_params[0] - true_a).abs() / true_a < tolerance,
                "A doesn't match: got {}, expected {}", optimized_params[0], true_a);
                
            assert!((optimized_params[1] - true_alpha).abs() / true_alpha < tolerance,
                "alpha doesn't match: got {}, expected {}", optimized_params[1], true_alpha);
                
            assert!((optimized_params[2] - true_b).abs() / true_b < tolerance,
                "B doesn't match: got {}, expected {}", optimized_params[2], true_b);
                
            assert!((optimized_params[3] - true_beta).abs() / true_beta < tolerance,
                "beta doesn't match: got {}, expected {}", optimized_params[3], true_beta);
            
            println!("Pharmacokinetic model test passed! Parameters: A = {}, alpha = {}, B = {}, beta = {}", 
                     optimized_params[0], optimized_params[1], optimized_params[2], optimized_params[3]);
        }
        LevenbergMarquardtResult::Error(err) => {
            panic!("LM optimization failed: {:?}", err);
        }
    }
}