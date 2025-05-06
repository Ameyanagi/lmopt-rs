use approx::assert_relative_eq;
use ndarray::{Array1, Array2, array, s};
use faer::{Mat, Col};
use lmopt_rs::utils::matrix_convert::{
    ndarray_to_faer, faer_to_ndarray,
    ndarray_vec_to_faer, faer_vec_to_ndarray,
};

// Test matrices of different sizes
const SMALL_DIMS: (usize, usize) = (2, 3);
const MEDIUM_DIMS: (usize, usize) = (10, 10);

// Helper function to create test matrices with deterministic values
fn create_test_matrix(rows: usize, cols: usize) -> Array2<f64> {
    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push((i * cols + j) as f64);
        }
    }
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

// Helper function to create test vectors with deterministic values
fn create_test_vector(size: usize) -> Array1<f64> {
    Array1::from_iter((0..size).map(|i| i as f64))
}

// ======== Matrix operations tests ========

#[test]
fn test_matrix_addition() {
    let (rows, cols) = SMALL_DIMS;
    let a = create_test_matrix(rows, cols);
    let b = create_test_matrix(rows, cols);
    
    // Add using ndarray
    let c_ndarray = &a + &b;
    
    // Add using faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    let c_faer = a_faer + b_faer;
    let c_from_faer = faer_to_ndarray(&c_faer).unwrap();
    
    // Results should match
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(c_ndarray[[i, j]], c_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_subtraction() {
    let (rows, cols) = SMALL_DIMS;
    let a = create_test_matrix(rows, cols);
    let b = create_test_matrix(rows, cols);
    
    // Subtract using ndarray
    let c_ndarray = &a - &b;
    
    // Subtract using faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    let c_faer = a_faer - b_faer;
    let c_from_faer = faer_to_ndarray(&c_faer).unwrap();
    
    // Results should match
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(c_ndarray[[i, j]], c_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_multiplication() {
    let (rows, cols) = SMALL_DIMS;
    let a = create_test_matrix(rows, cols);
    let b = create_test_matrix(cols, rows); // Note the transposed dimensions for multiplication
    
    // Multiply using ndarray
    let c_ndarray = a.dot(&b);
    
    // Multiply using faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    let c_faer = a_faer.mul(&b_faer);
    let c_from_faer = faer_to_ndarray(&c_faer).unwrap();
    
    // Results should match
    for i in 0..rows {
        for j in 0..rows {  // Note: result is rows x rows
            assert_relative_eq!(c_ndarray[[i, j]], c_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_vector_multiplication() {
    let (rows, cols) = SMALL_DIMS;
    let a = create_test_matrix(rows, cols);
    let v = create_test_vector(cols);
    
    // Multiply using ndarray
    let w_ndarray = a.dot(&v);
    
    // Multiply using faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let v_faer = ndarray_vec_to_faer(&v).unwrap();
    let w_faer = a_faer.mul(&v_faer);
    let w_from_faer = faer_vec_to_ndarray(&w_faer).unwrap();
    
    // Results should match
    for i in 0..rows {
        assert_relative_eq!(w_ndarray[i], w_from_faer[i], epsilon = 1e-10);
    }
}

#[test]
fn test_matrix_transposition() {
    let (rows, cols) = SMALL_DIMS;
    let a = create_test_matrix(rows, cols);
    
    // Transpose using ndarray
    let a_t_ndarray = a.t().to_owned();
    
    // Transpose using faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let a_t_faer = a_faer.transpose();
    let a_t_from_faer = faer_to_ndarray(&a_t_faer).unwrap();
    
    // Results should match
    for i in 0..cols {
        for j in 0..rows {
            assert_relative_eq!(a_t_ndarray[[i, j]], a_t_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_slicing() {
    let (rows, cols) = MEDIUM_DIMS;
    let a = create_test_matrix(rows, cols);
    
    // Slice using ndarray
    let a_slice_ndarray = a.slice(s![1..5, 2..7]).to_owned();
    
    // Convert to faer, then slice the converted result back to ndarray
    let a_faer = ndarray_to_faer(&a).unwrap();
    let a_from_faer = faer_to_ndarray(&a_faer).unwrap();
    let a_slice_from_faer = a_from_faer.slice(s![1..5, 2..7]).to_owned();
    
    // Results should match
    assert_eq!(a_slice_ndarray.shape(), a_slice_from_faer.shape());
    let (slice_rows, slice_cols) = (a_slice_ndarray.nrows(), a_slice_ndarray.ncols());
    for i in 0..slice_rows {
        for j in 0..slice_cols {
            assert_relative_eq!(a_slice_ndarray[[i, j]], a_slice_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

// ======== Matrix decomposition tests ========

#[test]
fn test_qr_decomposition() {
    let (rows, cols) = MEDIUM_DIMS;
    let a = create_test_matrix(rows, cols);
    
    // Convert to faer for QR decomposition
    let a_faer = ndarray_to_faer(&a).unwrap();
    
    // Compute QR decomposition
    let qr = faer::linalg::qr::QR::compute(a_faer.clone(), false);
    
    // Extract Q and R matrices
    let q = qr.q_matrix();
    let r = qr.r_matrix();
    
    // Convert back to ndarray
    let q_ndarray = faer_to_ndarray(&q).unwrap();
    let r_ndarray = faer_to_ndarray(&r).unwrap();
    
    // Verify that Q*R = A (approximately)
    let a_reconstructed = q_ndarray.dot(&r_ndarray);
    
    for i in 0..rows {
        for j in 0..cols {
            // Allow a slightly larger epsilon for numerical stability
            assert_relative_eq!(a[[i, j]], a_reconstructed[[i, j]], epsilon = 1e-8);
        }
    }
    
    // Verify that Q is orthogonal (Q^T * Q = I)
    let q_t = q_ndarray.t();
    let identity = q_t.dot(&q_ndarray);
    
    for i in 0..rows {
        for j in 0..rows {
            if i == j {
                assert_relative_eq!(identity[[i, j]], 1.0, epsilon = 1e-8);
            } else {
                assert_relative_eq!(identity[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }
}

#[test]
fn test_solve_linear_system() {
    // Create a simple linear system Ax = b
    let a = array![
        [2.0, 1.0, 1.0],
        [1.0, 3.0, 2.0],
        [1.0, 0.0, 0.0]
    ];
    let b = array![4.0, 5.0, 6.0];
    
    // Convert to faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_vec_to_faer(&b).unwrap();
    
    // Solve the system using QR decomposition
    let qr = faer::linalg::qr::QR::compute(a_faer.clone(), false);
    let x_faer = qr.solve(&b_faer).unwrap();
    
    // Convert solution back to ndarray
    let x = faer_vec_to_ndarray(&x_faer).unwrap();
    
    // Verify the solution: Ax = b
    let b_check = a.dot(&x);
    
    for i in 0..b.len() {
        assert_relative_eq!(b[i], b_check[i], epsilon = 1e-8);
    }
}

// ======== Specialized matrix operations for LM algorithm ========

#[test]
fn test_jtj_calculation() {
    // Create a Jacobian matrix J
    let (rows, cols) = (5, 3);  // 5 residuals, 3 parameters
    let j = create_test_matrix(rows, cols);
    
    // Calculate J^T * J using ndarray
    let j_t = j.t();
    let jtj_ndarray = j_t.dot(&j);
    
    // Calculate J^T * J using faer
    let j_faer = ndarray_to_faer(&j).unwrap();
    let j_t_faer = j_faer.transpose();
    let jtj_faer = j_t_faer.mul(&j_faer);
    let jtj_from_faer = faer_to_ndarray(&jtj_faer).unwrap();
    
    // Results should match
    for i in 0..cols {
        for j in 0..cols {
            assert_relative_eq!(jtj_ndarray[[i, j]], jtj_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_damping_matrix() {
    // Create a JTJ matrix
    let (params, _) = (3, 3);
    let jtj = create_test_matrix(params, params);
    
    // Get diagonal for damping
    let mut diag = Vec::with_capacity(params);
    for i in 0..params {
        diag.push(jtj[[i, i]]);
    }
    
    // Create augmented matrix with damping
    let lambda = 0.1;
    let mut augmented = jtj.clone();
    for i in 0..params {
        augmented[[i, i]] += lambda * diag[i];
    }
    
    // Convert to faer, apply damping, convert back
    let jtj_faer = ndarray_to_faer(&jtj).unwrap();
    let diag_faer = Mat::from_fn(params, params, |i, j| {
        if i == j {
            lambda * jtj_faer.get(i, i)
        } else {
            0.0
        }
    });
    let augmented_faer = jtj_faer + diag_faer;
    let augmented_from_faer = faer_to_ndarray(&augmented_faer).unwrap();
    
    // Results should match
    for i in 0..params {
        for j in 0..params {
            assert_relative_eq!(augmented[[i, j]], augmented_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

// ======== Property-based tests ========

// Here we test mathematical properties that should hold regardless of the matrix

#[test]
fn test_matrix_transpose_property() {
    // The transpose of a transpose should be the original matrix
    let (rows, cols) = MEDIUM_DIMS;
    let a = create_test_matrix(rows, cols);
    
    // Convert to faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    
    // Double transpose
    let a_t_t_faer = a_faer.transpose().transpose();
    
    // Convert back to ndarray
    let a_from_faer = faer_to_ndarray(&a_t_t_faer).unwrap();
    
    // Should be identical to the original
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(a[[i, j]], a_from_faer[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_addition_commutative() {
    // Matrix addition should be commutative: A + B = B + A
    let (rows, cols) = MEDIUM_DIMS;
    let a = create_test_matrix(rows, cols);
    let b = create_test_matrix(rows, cols).mapv(|x| x * 2.0); // Different values
    
    // Convert to faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    
    // Add in both orders
    let a_plus_b = a_faer + b_faer.clone();
    let b_plus_a = b_faer + a_faer;
    
    // Convert back and check equality
    let a_plus_b_ndarray = faer_to_ndarray(&a_plus_b).unwrap();
    let b_plus_a_ndarray = faer_to_ndarray(&b_plus_a).unwrap();
    
    for i in 0..rows {
        for j in 0..cols {
            assert_relative_eq!(a_plus_b_ndarray[[i, j]], b_plus_a_ndarray[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_matrix_multiplication_associative() {
    // Matrix multiplication should be associative: (AB)C = A(BC)
    let a = create_test_matrix(4, 3);
    let b = create_test_matrix(3, 2);
    let c = create_test_matrix(2, 5);
    
    // Convert to faer
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    let c_faer = ndarray_to_faer(&c).unwrap();
    
    // Multiply in two different orders
    let ab_faer = a_faer.mul(&b_faer);
    let abc_1 = ab_faer.mul(&c_faer);
    
    let bc_faer = b_faer.mul(&c_faer);
    let abc_2 = a_faer.mul(&bc_faer);
    
    // Convert back and check equality
    let abc_1_ndarray = faer_to_ndarray(&abc_1).unwrap();
    let abc_2_ndarray = faer_to_ndarray(&abc_2).unwrap();
    
    for i in 0..4 {
        for j in 0..5 {
            assert_relative_eq!(abc_1_ndarray[[i, j]], abc_2_ndarray[[i, j]], epsilon = 1e-10);
        }
    }
}

// ======== Performance benchmarks (to be moved to benches crate) ========

#[test]
fn test_large_matrix_operations_performance() {
    let (rows, cols) = (500, 500);
    let a = create_test_matrix(rows, cols);
    let b = create_test_matrix(rows, cols);
    
    // Time matrix addition
    let start = std::time::Instant::now();
    let _ = &a + &b;
    let ndarray_add_duration = start.elapsed();
    
    let a_faer = ndarray_to_faer(&a).unwrap();
    let b_faer = ndarray_to_faer(&b).unwrap();
    
    let start = std::time::Instant::now();
    let _ = a_faer.clone() + b_faer.clone();
    let faer_add_duration = start.elapsed();
    
    println!("ndarray matrix addition time: {:?}", ndarray_add_duration);
    println!("faer matrix addition time: {:?}", faer_add_duration);
    
    // Time matrix multiplication
    let start = std::time::Instant::now();
    let _ = a.dot(&b.t());
    let ndarray_mul_duration = start.elapsed();
    
    let start = std::time::Instant::now();
    let _ = a_faer.mul(&b_faer.transpose());
    let faer_mul_duration = start.elapsed();
    
    println!("ndarray matrix multiplication time: {:?}", ndarray_mul_duration);
    println!("faer matrix multiplication time: {:?}", faer_mul_duration);
    
    // These are informational only - no hard assertions
}