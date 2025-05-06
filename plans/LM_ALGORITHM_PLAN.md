# Levenberg-Marquardt Algorithm Implementation Plan

This document outlines the detailed plan for implementing the Levenberg-Marquardt (LM) algorithm in the lmopt-rs library, including mathematical foundations, technical approach, and implementation steps.

## Mathematical Background

The Levenberg-Marquardt algorithm is a numerical optimization technique for solving nonlinear least squares problems. It combines the Gauss-Newton algorithm and the gradient descent method, providing a good balance between the two.

### Objective Function

For a given problem with parameters β, the objective function to minimize is:

$$S(β) = \sum_{i=1}^{m} [r_i(β)]^2 = \mathbf{r}(β)^T \mathbf{r}(β)$$

where $\mathbf{r}(β)$ is the vector of residuals.

### Basic LM Update Rule

At each iteration, the parameter update is computed as:

$$δ = (\mathbf{J}^T\mathbf{J} + λ\mathbf{D})^{-1} \mathbf{J}^T \mathbf{r}$$

where:
- $\mathbf{J}$ is the Jacobian matrix of residuals with respect to parameters
- $λ$ is the damping parameter (Levenberg parameter)
- $\mathbf{D}$ is typically a diagonal matrix (often the diagonal of $\mathbf{J}^T\mathbf{J}$ or the identity matrix)

### Trust Region Interpretation

The LM algorithm can be viewed as a trust region method where $λ$ controls the size of the trust region. As $λ$ increases, the step size decreases, and the algorithm approaches gradient descent. As $λ$ decreases, the algorithm approaches Gauss-Newton.

## Implementation Approach

Our implementation will use a trust region approach with adaptive damping parameter updates. It will leverage faer for efficient matrix operations internally while providing a clean ndarray-based public API.

### Key Components

1. **Trust Region Implementation**: Controls step size and direction
2. **Adaptive Damping**: Updates $λ$ based on the success of each step
3. **Linear Solver**: Solves $(J^TJ + λD)δ = J^Tr$ efficiently
4. **Convergence Criteria**: Determines when to stop the algorithm
5. **Jacobian Calculation**: Computes the Jacobian matrix efficiently

## Implementation Steps

### 1. Core Data Structures

#### 1.1. LmConfig Structure

```rust
pub struct LmConfig {
    // Convergence parameters
    pub ftol: f64,          // Relative tolerance for cost function change
    pub xtol: f64,          // Relative tolerance for parameter change
    pub gtol: f64,          // Gradient tolerance (infinity norm)
    
    // Trust region parameters
    pub initial_lambda: f64, // Initial damping parameter
    pub lambda_factor: f64,  // Factor to increase/decrease lambda
    pub min_lambda: f64,     // Minimum allowed lambda value
    pub max_lambda: f64,     // Maximum allowed lambda value
    
    // Algorithm control
    pub max_iterations: usize, // Maximum number of iterations
    pub scale_diag: bool,      // Whether to scale using the diagonal of J^TJ
    pub verbose: bool,         // Whether to print iteration information
}
```

#### 1.2. LmResult Structure

```rust
pub struct LmResult {
    pub parameters: Array1<f64>,       // Fitted parameter values
    pub cost: f64,                     // Final cost function value
    pub residuals: Array1<f64>,        // Final residuals
    pub jacobian: Option<Array2<f64>>, // Final Jacobian matrix
    pub iterations: usize,             // Number of iterations performed
    pub success: bool,                 // Whether the fit converged
    pub message: String,               // Description of the result
    pub covariance: Option<Array2<f64>>, // Covariance matrix (if calculated)
    pub lambda: f64,                   // Final lambda value
    pub history: Option<LmHistory>,    // Optimization history (if recorded)
}
```

#### 1.3. LmHistory Structure

```rust
pub struct LmHistory {
    pub parameters: Vec<Array1<f64>>,   // Parameter values at each iteration
    pub costs: Vec<f64>,                // Cost function values at each iteration
    pub lambdas: Vec<f64>,              // Lambda values at each iteration
    pub gradient_norms: Vec<f64>,       // Gradient norm at each iteration
    pub step_norms: Vec<f64>,           // Step size at each iteration
}
```

### 2. Main Algorithm Implementation

#### 2.1. LevenbergMarquardt Struct

```rust
pub struct LevenbergMarquardt {
    config: LmConfig,
    record_history: bool,
}
```

#### 2.2. Minimization Method

```rust
impl LevenbergMarquardt {
    pub fn minimize<P: Problem>(&self, problem: &P, initial_params: Array1<f64>) -> Result<LmResult> {
        // Initialize variables
        let mut params = initial_params;
        let mut lambda = self.config.initial_lambda;
        let mut iter = 0;
        let mut history = self.init_history();
        
        // Evaluate initial residuals and cost
        let residuals = problem.eval(&params)?;
        let mut cost = self.calculate_cost(&residuals);
        
        // Main iteration loop
        while iter < self.config.max_iterations {
            // Compute Jacobian
            let jacobian = problem.jacobian(&params)?;
            
            // Compute gradient J^T * r
            let gradient = self.compute_gradient(&jacobian, &residuals);
            
            // Check gradient convergence
            if self.check_gradient_convergence(&gradient) {
                return Ok(self.create_result(params, cost, residuals, Some(jacobian), 
                    iter, true, "Gradient convergence", lambda, history));
            }
            
            // Compute JTJ and scaling diagonal if needed
            let jtj = self.compute_jtj(&jacobian);
            let diag = self.compute_scaling_diagonal(&jtj);
            
            // Compute step using trust region approach
            let step = self.compute_step(&jacobian, &residuals, &jtj, &diag, lambda)?;
            
            // Check step convergence
            if self.check_step_convergence(&step) {
                return Ok(self.create_result(params, cost, residuals, Some(jacobian), 
                    iter, true, "Step size convergence", lambda, history));
            }
            
            // Try the step
            let new_params = &params + &step;
            let new_residuals = problem.eval(&new_params)?;
            let new_cost = self.calculate_cost(&new_residuals);
            
            // Compute actual vs. predicted reduction
            let predicted_reduction = self.compute_predicted_reduction(&step, &gradient, &jtj, &diag, lambda);
            let actual_reduction = cost - new_cost;
            let rho = actual_reduction / predicted_reduction;
            
            // Update lambda based on step success
            lambda = self.update_lambda(lambda, rho);
            
            // Accept or reject step
            if rho > 0.0 {
                // Accept step
                params = new_params;
                residuals = new_residuals;
                cost = new_cost;
                
                // Check function value convergence
                if self.check_cost_convergence(cost, new_cost) {
                    return Ok(self.create_result(params, cost, residuals, Some(jacobian), 
                        iter, true, "Cost function convergence", lambda, history));
                }
            }
            
            // Update history
            if self.record_history {
                self.update_history(&mut history, &params, cost, lambda, 
                    gradient.iter().map(|x| x.abs()).fold(0.0, f64::max), 
                    step.iter().map(|x| x*x).sum::<f64>().sqrt());
            }
            
            iter += 1;
        }
        
        // Max iterations reached
        Ok(self.create_result(params, cost, residuals, None, iter, false, 
            "Maximum iterations reached", lambda, history))
    }
    
    // Helper methods...
}
```

### 3. Key Algorithm Components

#### 3.1. Step Calculation

```rust
fn compute_step(&self, 
    jacobian: &Array2<f64>, 
    residuals: &Array1<f64>, 
    jtj: &Array2<f64>, 
    diag: &Array1<f64>, 
    lambda: f64) -> Result<Array1<f64>> {
    
    // Create augmented matrix (J^TJ + λD)
    let n_params = jtj.shape()[0];
    let mut augmented = jtj.clone();
    
    // Add λD to the diagonal
    for i in 0..n_params {
        augmented[[i, i]] += lambda * diag[i];
    }
    
    // Compute right-hand side (-J^T * r)
    let rhs = -1.0 * jacobian.t().dot(residuals);
    
    // Solve the linear system (J^TJ + λD)δ = -J^Tr using faer
    let augmented_faer = ndarray_to_faer(&augmented)?;
    let rhs_faer = ndarray_vec_to_faer(&rhs)?;
    
    // Use QR decomposition for stability
    let qr = faer::linalg::qr::QR::compute(augmented_faer.clone(), false);
    let step_faer = qr.solve(&rhs_faer)?;
    
    // Convert back to ndarray
    let step = faer_vec_to_ndarray(&step_faer)?;
    
    Ok(step)
}
```

#### 3.2. Lambda Update

```rust
fn update_lambda(&self, lambda: f64, rho: f64) -> f64 {
    let lambda_factor = self.config.lambda_factor;
    
    let new_lambda = if rho > 0.75 {
        // Good step - decrease lambda (move towards Gauss-Newton)
        (lambda / lambda_factor).max(self.config.min_lambda)
    } else if rho < 0.25 {
        // Poor step - increase lambda (move towards gradient descent)
        (lambda * lambda_factor).min(self.config.max_lambda)
    } else {
        // Reasonable step - keep lambda the same
        lambda
    };
    
    new_lambda
}
```

#### 3.3. Convergence Checks

```rust
fn check_gradient_convergence(&self, gradient: &Array1<f64>) -> bool {
    // Check if the infinity norm of the gradient is below the threshold
    let g_norm = gradient.iter().map(|x| x.abs()).fold(0.0, f64::max);
    g_norm < self.config.gtol
}

fn check_step_convergence(&self, step: &Array1<f64>) -> bool {
    // Check if the relative step size is below the threshold
    let step_norm = step.iter().map(|x| x*x).sum::<f64>().sqrt();
    let params_norm = params.iter().map(|x| x*x).sum::<f64>().sqrt();
    
    step_norm < self.config.xtol * (params_norm + self.config.xtol)
}

fn check_cost_convergence(&self, old_cost: f64, new_cost: f64) -> bool {
    // Check if the relative change in cost is below the threshold
    (old_cost - new_cost).abs() < self.config.ftol * old_cost
}
```

#### 3.4. Predicted Reduction Calculation

```rust
fn compute_predicted_reduction(
    &self, 
    step: &Array1<f64>, 
    gradient: &Array1<f64>, 
    jtj: &Array2<f64>, 
    diag: &Array1<f64>, 
    lambda: f64) -> f64 {
    
    // Compute 0.5 * step^T * (gradient + lambda * D * step)
    let step_dot_grad = step.dot(gradient);
    
    let scaled_step = Array1::from_iter(
        step.iter().zip(diag.iter()).map(|(s, d)| s * d * lambda)
    );
    
    let step_dot_scaled = step.dot(&scaled_step);
    
    0.5 * (step_dot_grad + step_dot_scaled)
}
```

### 4. Test Problems

We'll implement several test problems to validate our LM implementation:

1. **Rosenbrock Function**: A classic test function with a narrow valley
2. **Gaussian Peaks**: Fitting sum of Gaussian functions to data
3. **Exponential Decay**: Fitting exponential decay curves
4. **Linear Regression**: As a simple sanity check
5. **NIST Test Problems**: Standard reference datasets from NIST

### 5. Validation against Reference Implementations

To ensure correctness, we'll compare our results against:

1. **levenberg-marquardt crate**: For compatibility and correctness
2. **lmfit-py**: For algorithm behavior and convergence
3. **Analytical solutions**: For problems with known solutions

### 6. Performance Optimization

- **Matrix Storage**: Use efficient storage format for Jacobian
- **Linear Solver**: Optimize linear solver for different problem sizes
- **Jacobian Calculation**: Use the most efficient method (autodiff, analytical, or numerical)
- **Memory Reuse**: Reuse allocated memory where possible
- **SIMD Operations**: Leverage SIMD for matrix operations via faer

## Timeline

| Task | Estimated Duration | Dependencies |
|------|-------------------|--------------|
| Core Data Structures | 2 days | None |
| Step Calculation | 3 days | Core Data Structures |
| Lambda Update | 1 day | Core Data Structures |
| Convergence Checks | 2 days | Core Data Structures |
| Main Algorithm Loop | 3 days | Step Calculation, Lambda Update, Convergence Checks |
| Test Problems | 2 days | None |
| Validation | 3 days | Main Algorithm Loop, Test Problems |
| Performance Optimization | 5 days | Main Algorithm Loop |

## Success Criteria

1. Algorithm converges to correct solutions for all test problems
2. Performance is comparable to or better than reference implementations
3. Algorithm handles edge cases gracefully (singular matrices, poorly conditioned problems)
4. API is clean and consistent with the rest of the library
5. Code is well-documented and tested
6. Validation results match reference implementations