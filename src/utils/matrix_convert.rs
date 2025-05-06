//! Matrix conversion utilities for the lmopt-rs library.
//!
//! This module provides functions to convert between different matrix types:
//! - ndarray (Array2, Array1)
//! - faer (Mat, Col)
//! - nalgebra (DMatrix, DVector)
//!
//! These conversions are needed to provide compatibility with the levenberg-marquardt
//! crate (which uses nalgebra) while leveraging faer for internal matrix operations.

use faer::{Mat, Col};
// use faer::dyn_stack::PodStack;
use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};
use crate::error::Result; // Removed LmOptError since it's not used

// Define a constraint for numeric types we can convert between
pub trait NumericScalar: Clone + Default + 'static { }

// Implement trait for common numeric types
impl NumericScalar for f32 { }
impl NumericScalar for f64 { }

// === ndarray <-> faer conversions ===

/// Convert an ndarray Array2 to a faer Mat.
///
/// # Arguments
///
/// * `arr` - The ndarray Array2 to convert
///
/// # Returns
///
/// * A faer Mat with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn ndarray_to_faer<T: NumericScalar>(arr: &Array2<T>) -> Result<Mat<T>> {
    // Get dimensions
    let rows = arr.nrows();
    let cols = arr.ncols();
    
    // Create a new faer matrix with default values
    let mut mat = Mat::from_fn(rows, cols, |_, _| T::default());
    
    // Copy data from ndarray to faer
    // Note: ndarray is row-major by default, faer is column-major
    for i in 0..rows {
        for j in 0..cols {
            *mat.get_mut(i, j) = arr[[i, j]].clone();
        }
    }
    
    Ok(mat)
}

/// Convert a faer Mat to an ndarray Array2.
///
/// # Arguments
///
/// * `mat` - The faer Mat or MatRef to convert
///
/// # Returns
///
/// * An ndarray Array2 with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn faer_to_ndarray<T: NumericScalar>(mat: &Mat<T>) -> Result<Array2<T>> {
    // Get dimensions
    let rows = mat.nrows();
    let cols = mat.ncols();
    
    // Create a new ndarray with the right shape
    let mut arr = Array2::from_elem((rows, cols), mat.get(0, 0).clone());
    
    // Copy data from faer to ndarray
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = mat.get(i, j).clone();
        }
    }
    
    Ok(arr)
}

/// Convert an ndarray Array1 to a faer Col (column vector).
///
/// # Arguments
///
/// * `arr` - The ndarray Array1 to convert
///
/// # Returns
///
/// * A faer Col with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn ndarray_vec_to_faer<T: NumericScalar>(arr: &Array1<T>) -> Result<Col<T>> {
    // Get dimension
    let n = arr.len();
    
    // Create a new faer column vector
    let mut col = Col::from_fn(n, |_| T::default());
    
    // Copy data from ndarray to faer
    for i in 0..n {
        *col.get_mut(i) = arr[i].clone();
    }
    
    Ok(col)
}

/// Convert a faer Col (column vector) to an ndarray Array1.
///
/// # Arguments
///
/// * `col` - The faer Col to convert
///
/// # Returns
///
/// * An ndarray Array1 with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn faer_vec_to_ndarray<T: NumericScalar>(col: &Col<T>) -> Result<Array1<T>> {
    // Get dimension
    let n = col.nrows();
    
    // Create a new ndarray vector
    let mut arr = Array1::from_elem(n, col.get(0).clone());
    
    // Copy data from faer to ndarray
    for i in 0..n {
        arr[i] = col.get(i).clone();
    }
    
    Ok(arr)
}

// === nalgebra <-> faer conversions ===

/// Convert a nalgebra DMatrix to a faer Mat.
///
/// # Arguments
///
/// * `mat` - The nalgebra DMatrix to convert
///
/// # Returns
///
/// * A faer Mat with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn nalgebra_to_faer<T: NumericScalar + nalgebra::Scalar>(mat: &DMatrix<T>) -> Result<Mat<T>> {
    // Get dimensions
    let rows = mat.nrows();
    let cols = mat.ncols();
    
    // Create a new faer matrix
    let mut faer_mat = Mat::from_fn(rows, cols, |_, _| T::default());
    
    // Copy data from nalgebra to faer
    for i in 0..rows {
        for j in 0..cols {
            *faer_mat.get_mut(i, j) = mat[(i, j)].clone();
        }
    }
    
    Ok(faer_mat)
}

/// Convert a faer Mat to a nalgebra DMatrix.
///
/// # Arguments
///
/// * `mat` - The faer Mat to convert
///
/// # Returns
///
/// * A nalgebra DMatrix with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn faer_to_nalgebra<T: NumericScalar + nalgebra::Scalar>(mat: &Mat<T>) -> Result<DMatrix<T>> {
    // Get dimensions
    let rows = mat.nrows();
    let cols = mat.ncols();
    
    // Create a new nalgebra matrix
    let mut nalgebra_mat = DMatrix::from_element(rows, cols, mat.get(0, 0).clone());
    
    // Copy data from faer to nalgebra
    for i in 0..rows {
        for j in 0..cols {
            nalgebra_mat[(i, j)] = mat.get(i, j).clone();
        }
    }
    
    Ok(nalgebra_mat)
}

/// Convert a nalgebra DVector to a faer Col.
///
/// # Arguments
///
/// * `vec` - The nalgebra DVector to convert
///
/// # Returns
///
/// * A faer Col with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn nalgebra_vec_to_faer<T: NumericScalar + nalgebra::Scalar>(vec: &DVector<T>) -> Result<Col<T>> {
    // Get dimension
    let n = vec.nrows();
    
    // Create a new faer column vector
    let mut col = Col::from_fn(n, |_| T::default());
    
    // Copy data from nalgebra to faer
    for i in 0..n {
        *col.get_mut(i) = vec[i].clone();
    }
    
    Ok(col)
}

/// Convert a faer Col to a nalgebra DVector.
///
/// # Arguments
///
/// * `col` - The faer Col to convert
///
/// # Returns
///
/// * A nalgebra DVector with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn faer_vec_to_nalgebra<T: NumericScalar + nalgebra::Scalar>(col: &Col<T>) -> Result<DVector<T>> {
    // Get dimension
    let n = col.nrows();
    
    // Create a new nalgebra vector
    let mut vec = DVector::from_element(n, col.get(0).clone());
    
    // Copy data from faer to nalgebra
    for i in 0..n {
        vec[i] = col.get(i).clone();
    }
    
    Ok(vec)
}

// === ndarray <-> nalgebra conversions ===

/// Convert an ndarray Array2 to a nalgebra DMatrix.
///
/// # Arguments
///
/// * `arr` - The ndarray Array2 to convert
///
/// # Returns
///
/// * A nalgebra DMatrix with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn ndarray_to_nalgebra<T: Clone + nalgebra::Scalar>(arr: &Array2<T>) -> Result<DMatrix<T>> {
    // Get dimensions
    let rows = arr.nrows();
    let cols = arr.ncols();
    
    // Create a new nalgebra matrix
    let mut mat = DMatrix::from_element(rows, cols, arr[[0, 0]].clone());
    
    // Copy data from ndarray to nalgebra
    for i in 0..rows {
        for j in 0..cols {
            mat[(i, j)] = arr[[i, j]].clone();
        }
    }
    
    Ok(mat)
}

/// Convert a nalgebra DMatrix to an ndarray Array2.
///
/// # Arguments
///
/// * `mat` - The nalgebra DMatrix to convert
///
/// # Returns
///
/// * An ndarray Array2 with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn nalgebra_to_ndarray<T: Clone>(mat: &DMatrix<T>) -> Result<Array2<T>> {
    // Get dimensions
    let rows = mat.nrows();
    let cols = mat.ncols();
    
    // Create a new ndarray with the right shape
    let mut arr = Array2::from_elem((rows, cols), mat[(0, 0)].clone());
    
    // Copy data from nalgebra to ndarray
    for i in 0..rows {
        for j in 0..cols {
            arr[[i, j]] = mat[(i, j)].clone();
        }
    }
    
    Ok(arr)
}

/// Convert an ndarray Array1 to a nalgebra DVector.
///
/// # Arguments
///
/// * `arr` - The ndarray Array1 to convert
///
/// # Returns
///
/// * A nalgebra DVector with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn ndarray_vec_to_nalgebra<T: Clone + nalgebra::Scalar>(arr: &Array1<T>) -> Result<DVector<T>> {
    // Get dimension
    let n = arr.len();
    
    // Create a new nalgebra vector
    let mut vec = DVector::from_element(n, arr[0].clone());
    
    // Copy data from ndarray to nalgebra
    for i in 0..n {
        vec[i] = arr[i].clone();
    }
    
    Ok(vec)
}

/// Convert a nalgebra DVector to an ndarray Array1.
///
/// # Arguments
///
/// * `vec` - The nalgebra DVector to convert
///
/// # Returns
///
/// * An ndarray Array1 with the same data
///
/// # Errors
///
/// * `LmOptError::ConversionError` if the conversion fails
pub fn nalgebra_vec_to_ndarray<T: Clone>(vec: &DVector<T>) -> Result<Array1<T>> {
    // Get dimension
    let n = vec.nrows();
    
    // Create a new ndarray vector
    let mut arr = Array1::from_elem(n, vec[0].clone());
    
    // Copy data from nalgebra to ndarray
    for i in 0..n {
        arr[i] = vec[i].clone();
    }
    
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_ndarray_faer_roundtrip_f64() {
        // Create a test matrix
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        // Convert to faer and back
        let faer_mat = ndarray_to_faer(&arr).unwrap();
        let arr2 = faer_to_ndarray(&faer_mat).unwrap();
        
        // Check dimensions
        assert_eq!(arr.shape(), arr2.shape());
        
        // Check values
        for i in 0..arr.nrows() {
            for j in 0..arr.ncols() {
                assert_relative_eq!(arr[[i, j]], arr2[[i, j]]);
            }
        }
    }
    
    #[test]
    fn test_ndarray_vec_faer_vec_roundtrip_f64() {
        // Create a test vector
        let arr = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        // Convert to faer and back
        let faer_col = ndarray_vec_to_faer(&arr).unwrap();
        let arr2 = faer_vec_to_ndarray(&faer_col).unwrap();
        
        // Check dimensions
        assert_eq!(arr.len(), arr2.len());
        
        // Check values
        for i in 0..arr.len() {
            assert_relative_eq!(arr[i], arr2[i]);
        }
    }
    
    #[test]
    fn test_nalgebra_faer_roundtrip_f64() {
        // Create a test matrix
        let mat = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        
        // Convert to faer and back
        let faer_mat = nalgebra_to_faer(&mat).unwrap();
        let mat2 = faer_to_nalgebra(&faer_mat).unwrap();
        
        // Check dimensions
        assert_eq!(mat.nrows(), mat2.nrows());
        assert_eq!(mat.ncols(), mat2.ncols());
        
        // Check values
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                assert_relative_eq!(mat[(i, j)], mat2[(i, j)]);
            }
        }
    }
    
    #[test]
    fn test_ndarray_nalgebra_roundtrip_f64() {
        // Create a test matrix
        let arr = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        // Convert to nalgebra and back
        let nalgebra_mat = ndarray_to_nalgebra(&arr).unwrap();
        let arr2 = nalgebra_to_ndarray(&nalgebra_mat).unwrap();
        
        // Check dimensions
        assert_eq!(arr.shape(), arr2.shape());
        
        // Check values
        for i in 0..arr.nrows() {
            for j in 0..arr.ncols() {
                assert_relative_eq!(arr[[i, j]], arr2[[i, j]]);
            }
        }
    }
    
    #[test]
    fn test_matrix_dimensions() {
        // Test various dimensions
        for (rows, cols) in [(1, 1), (5, 5), (10, 20), (100, 1), (1, 100)].iter() {
            // Create test matrices
            let mut arr_data = Vec::new();
            for i in 0..*rows {
                for j in 0..*cols {
                    arr_data.push((i * *cols + j) as f64);
                }
            }
            
            let arr = Array2::from_shape_vec((*rows, *cols), arr_data).unwrap();
            
            // Test conversions
            let faer_mat = ndarray_to_faer(&arr).unwrap();
            let nalgebra_mat = ndarray_to_nalgebra(&arr).unwrap();
            
            // Check dimensions
            assert_eq!(faer_mat.nrows(), *rows);
            assert_eq!(faer_mat.ncols(), *cols);
            assert_eq!(nalgebra_mat.nrows(), *rows);
            assert_eq!(nalgebra_mat.ncols(), *cols);
        }
    }
}