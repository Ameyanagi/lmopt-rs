//! Integration tests for matrix conversion utilities.

use approx::assert_relative_eq;
use lmopt_rs::utils::{
    faer_to_nalgebra, faer_to_ndarray, faer_vec_to_nalgebra, faer_vec_to_ndarray, nalgebra_to_faer,
    nalgebra_to_ndarray, nalgebra_vec_to_faer, nalgebra_vec_to_ndarray, ndarray_to_faer,
    ndarray_to_nalgebra, ndarray_vec_to_faer, ndarray_vec_to_nalgebra,
};
use nalgebra::{DMatrix, DVector};
use ndarray::{array, Array1, Array2};

#[test]
fn test_ndarray_faer_roundtrip_2d_f64() {
    // Create a test matrix with known values
    let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Convert to faer and back
    let faer_mat = ndarray_to_faer(&arr).unwrap();
    let arr2 = faer_to_ndarray(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(arr.shape(), arr2.shape());

    // Check values
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            assert_relative_eq!(arr[[i, j]], arr2[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_ndarray_faer_roundtrip_1d_f64() {
    // Create a test vector with known values
    let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Convert to faer and back
    let faer_col = ndarray_vec_to_faer(&arr).unwrap();
    let arr2 = faer_vec_to_ndarray(&faer_col).unwrap();

    // Check dimensions
    assert_eq!(arr.len(), arr2.len());

    // Check values
    for i in 0..arr.len() {
        assert_relative_eq!(arr[i], arr2[i], epsilon = 1e-10);
    }
}

#[test]
fn test_nalgebra_faer_roundtrip_2d_f64() {
    // Create a test matrix with known values
    let mat = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Convert to faer and back
    let faer_mat = nalgebra_to_faer(&mat).unwrap();
    let mat2 = faer_to_nalgebra(&faer_mat).unwrap();

    // Check dimensions
    assert_eq!(mat.nrows(), mat2.nrows());
    assert_eq!(mat.ncols(), mat2.ncols());

    // Check values
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            assert_relative_eq!(mat[(i, j)], mat2[(i, j)], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_nalgebra_faer_roundtrip_1d_f64() {
    // Create a test vector with known values
    let vec = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    // Convert to faer and back
    let faer_col = nalgebra_vec_to_faer(&vec).unwrap();
    let vec2 = faer_vec_to_nalgebra(&faer_col).unwrap();

    // Check dimensions
    assert_eq!(vec.nrows(), vec2.nrows());

    // Check values
    for i in 0..vec.nrows() {
        assert_relative_eq!(vec[i], vec2[i], epsilon = 1e-10);
    }
}

#[test]
fn test_ndarray_nalgebra_roundtrip_2d_f64() {
    // Create a test matrix with known values
    let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Convert to nalgebra and back
    let mat = ndarray_to_nalgebra(&arr).unwrap();
    let arr2 = nalgebra_to_ndarray(&mat).unwrap();

    // Check dimensions
    assert_eq!(arr.shape(), arr2.shape());

    // Check values
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            assert_relative_eq!(arr[[i, j]], arr2[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_ndarray_nalgebra_roundtrip_1d_f64() {
    // Create a test vector with known values
    let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Convert to nalgebra and back
    let vec = ndarray_vec_to_nalgebra(&arr).unwrap();
    let arr2 = nalgebra_vec_to_ndarray(&vec).unwrap();

    // Check dimensions
    assert_eq!(arr.len(), arr2.len());

    // Check values
    for i in 0..arr.len() {
        assert_relative_eq!(arr[i], arr2[i], epsilon = 1e-10);
    }
}

#[test]
fn test_matrix_dimensions() {
    // Test various dimensions
    for (rows, cols) in [(1, 1), (5, 5), (10, 20), (100, 1), (1, 100)].iter() {
        // Create an ndarray matrix
        let mut arr_data = Vec::new();
        for i in 0..*rows {
            for j in 0..*cols {
                arr_data.push((i * *cols + j) as f64);
            }
        }

        let arr = Array2::from_shape_vec((*rows, *cols), arr_data).unwrap();

        // Test ndarray -> faer conversion
        let faer_mat = ndarray_to_faer(&arr).unwrap();
        assert_eq!(faer_mat.nrows(), *rows);
        assert_eq!(faer_mat.ncols(), *cols);

        // Test ndarray -> nalgebra conversion
        let mat = ndarray_to_nalgebra(&arr).unwrap();
        assert_eq!(mat.nrows(), *rows);
        assert_eq!(mat.ncols(), *cols);
    }
}

#[test]
fn test_edge_cases() {
    // Test empty matrix
    {
        let arr = Array2::<f64>::from_shape_vec((0, 0), vec![]).unwrap();
        let faer_mat = ndarray_to_faer(&arr).unwrap();
        assert_eq!(faer_mat.nrows(), 0);
        assert_eq!(faer_mat.ncols(), 0);
    }

    // Test single element matrix
    {
        let arr = Array2::from_shape_vec((1, 1), vec![42.0]).unwrap();
        let faer_mat = ndarray_to_faer(&arr).unwrap();
        assert_eq!(faer_mat.nrows(), 1);
        assert_eq!(faer_mat.ncols(), 1);
        let value = *&faer_mat.get(0, 0);
        assert!((value - 42.0f64).abs() < 1e-10);
    }

    // Test empty vector
    {
        let arr = Array1::<f64>::from_vec(vec![]);
        let faer_col = ndarray_vec_to_faer(&arr).unwrap();
        assert_eq!(faer_col.nrows(), 0);
    }

    // Test single element vector
    {
        let arr = Array1::from_vec(vec![42.0]);
        let faer_col = ndarray_vec_to_faer(&arr).unwrap();
        assert_eq!(faer_col.nrows(), 1);
        let value = *&faer_col.get(0);
        assert!((value - 42.0f64).abs() < 1e-10);
    }
}

#[test]
fn test_three_way_conversion() {
    // Create a test matrix with known values
    let arr = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Convert ndarray -> faer -> nalgebra -> ndarray
    let faer_mat = ndarray_to_faer(&arr).unwrap();
    let mat = faer_to_nalgebra(&faer_mat).unwrap();
    let arr2 = nalgebra_to_ndarray(&mat).unwrap();

    // Check dimensions
    assert_eq!(arr.shape(), arr2.shape());

    // Check values
    for i in 0..arr.nrows() {
        for j in 0..arr.ncols() {
            assert_relative_eq!(arr[[i, j]], arr2[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_precision_preservation() {
    // Create a matrix with very small and very large values
    let values = vec![1e-10, 1e10, -1e-10, -1e10, 0.0];
    let arr = Array1::from_vec(values.clone());

    // Test ndarray -> faer -> ndarray
    let faer_col = ndarray_vec_to_faer(&arr).unwrap();
    let arr2 = faer_vec_to_ndarray(&faer_col).unwrap();

    for i in 0..values.len() {
        assert_relative_eq!(values[i], arr2[i], epsilon = 1e-15, max_relative = 1e-10);
    }

    // Test ndarray -> nalgebra -> ndarray
    let vec = ndarray_vec_to_nalgebra(&arr).unwrap();
    let arr3 = nalgebra_vec_to_ndarray(&vec).unwrap();

    for i in 0..values.len() {
        assert_relative_eq!(values[i], arr3[i], epsilon = 1e-15, max_relative = 1e-10);
    }
}

#[test]
fn test_special_values() {
    // Create a matrix with special values (NaN, Inf)
    let values = vec![std::f64::NAN, std::f64::INFINITY, std::f64::NEG_INFINITY];
    let arr = Array1::from_vec(values);

    // Test ndarray -> nalgebra -> ndarray for special values
    let vec = ndarray_vec_to_nalgebra(&arr).unwrap();
    let arr2 = nalgebra_vec_to_ndarray(&vec).unwrap();

    assert!(arr2[0].is_nan());
    assert!(arr2[1].is_infinite() && arr2[1] > 0.0);
    assert!(arr2[2].is_infinite() && arr2[2] < 0.0);
}

#[test]
fn test_large_matrix() {
    // Create a large matrix (1000x1000) to test performance
    let n = 100; // Using 100x100 for testing; increase for benchmarking
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data.push((i * n + j) as f64);
        }
    }

    let arr = Array2::from_shape_vec((n, n), data).unwrap();

    // Test ndarray -> faer -> ndarray
    let faer_mat = ndarray_to_faer(&arr).unwrap();
    let arr2 = faer_to_ndarray(&faer_mat).unwrap();

    // Check a few sample values
    assert_relative_eq!(arr[[0, 0]], arr2[[0, 0]], epsilon = 1e-10);
    assert_relative_eq!(arr[[n / 2, n / 2]], arr2[[n / 2, n / 2]], epsilon = 1e-10);
    assert_relative_eq!(arr[[n - 1, n - 1]], arr2[[n - 1, n - 1]], epsilon = 1e-10);
}
