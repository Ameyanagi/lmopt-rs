//! Benchmarks for matrix operations
//!
//! This file contains benchmarks for matrix operations used in the
//! Levenberg-Marquardt algorithm, comparing different implementations.

#![feature(test)]
extern crate test;

use faer::{mat, Col, Mat, MatRef};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use test::Bencher;

// Import utility functions for matrix conversions
use lmopt_rs::utils::matrix_convert::{
    faer_to_ndarray, nalgebra_to_ndarray, ndarray_to_faer, ndarray_to_nalgebra,
};

/// Benchmark matrix multiplication with ndarray
#[bench]
fn bench_ndarray_matmul_100x100(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));
    let b = Array2::<f64>::ones((100, 100));

    b.iter(|| test::black_box(a.dot(&b)));
}

/// Benchmark matrix multiplication with faer
#[bench]
fn bench_faer_matmul_100x100(b: &mut Bencher) {
    let a = Mat::<f64>::ones(100, 100);
    let b = Mat::<f64>::ones(100, 100);

    b.iter(|| test::black_box(&a * &b));
}

/// Benchmark matrix multiplication with nalgebra
#[bench]
fn bench_nalgebra_matmul_100x100(b: &mut Bencher) {
    let a = DMatrix::<f64>::repeat(100, 100, 1.0);
    let b = DMatrix::<f64>::repeat(100, 100, 1.0);

    b.iter(|| test::black_box(&a * &b));
}

/// Benchmark QR decomposition with ndarray_linalg
#[bench]
fn bench_ndarray_qr_100x50(b: &mut Bencher) {
    use ndarray_linalg::QR;

    let a = Array2::<f64>::ones((100, 50));

    b.iter(|| test::black_box(a.qr().unwrap()));
}

/// Benchmark QR decomposition with faer
#[bench]
fn bench_faer_qr_100x50(b: &mut Bencher) {
    let a = Mat::<f64>::ones(100, 50);

    b.iter(|| {
        let mut q = Mat::<f64>::zeros(100, 50);
        let mut r = Mat::<f64>::zeros(50, 50);
        test::black_box(faer::linalg::householder::qr::compute(
            a.as_ref(),
            q.as_mut(),
            r.as_mut(),
            None,
            faer::Side::LEFT,
        ));
    });
}

/// Benchmark QR decomposition with nalgebra
#[bench]
fn bench_nalgebra_qr_100x50(b: &mut Bencher) {
    use nalgebra::QR;

    let a = DMatrix::<f64>::repeat(100, 50, 1.0);

    b.iter(|| test::black_box(a.qr()));
}

/// Benchmark matrix conversion from ndarray to faer
#[bench]
fn bench_ndarray_to_faer_100x100(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));

    b.iter(|| test::black_box(ndarray_to_faer(&a)));
}

/// Benchmark matrix conversion from faer to ndarray
#[bench]
fn bench_faer_to_ndarray_100x100(b: &mut Bencher) {
    let a = Mat::<f64>::ones(100, 100);

    b.iter(|| test::black_box(faer_to_ndarray(&a)));
}

/// Benchmark matrix conversion from ndarray to nalgebra
#[bench]
fn bench_ndarray_to_nalgebra_100x100(b: &mut Bencher) {
    let a = Array2::<f64>::ones((100, 100));

    b.iter(|| test::black_box(ndarray_to_nalgebra(&a)));
}

/// Benchmark matrix conversion from nalgebra to ndarray
#[bench]
fn bench_nalgebra_to_ndarray_100x100(b: &mut Bencher) {
    let a = DMatrix::<f64>::repeat(100, 100, 1.0);

    b.iter(|| test::black_box(nalgebra_to_ndarray(&a)));
}

/// Benchmark vector operations with ndarray
#[bench]
fn bench_ndarray_vector_ops_1000(b: &mut Bencher) {
    let a = Array1::<f64>::ones(1000);
    let b = Array1::<f64>::ones(1000);

    b.iter(|| {
        // Vector addition, dot product, and norm
        let c = &a + &b;
        let dot = a.dot(&b);
        let norm = c.dot(&c).sqrt();
        test::black_box((dot, norm))
    });
}

/// Benchmark vector operations with faer
#[bench]
fn bench_faer_vector_ops_1000(b: &mut Bencher) {
    let a = Mat::<f64>::ones(1000, 1);
    let b = Mat::<f64>::ones(1000, 1);

    b.iter(|| {
        // Vector addition, dot product, and norm
        let c = &a + &b;
        let dot = a.as_ref().transpose() * b.as_ref();
        let norm = (c.as_ref().transpose() * c.as_ref()).get(0, 0).sqrt();
        test::black_box((dot.get(0, 0), norm))
    });
}

/// Benchmark vector operations with nalgebra
#[bench]
fn bench_nalgebra_vector_ops_1000(b: &mut Bencher) {
    let a = DVector::<f64>::repeat(1000, 1.0);
    let b = DVector::<f64>::repeat(1000, 1.0);

    b.iter(|| {
        // Vector addition, dot product, and norm
        let c = &a + &b;
        let dot = a.dot(&b);
        let norm = c.norm();
        test::black_box((dot, norm))
    });
}

/// Benchmark solving a linear system with ndarray_linalg
#[bench]
fn bench_ndarray_solve_100x100(b: &mut Bencher) {
    use ndarray_linalg::Solve;

    // Create a matrix and vector
    let mut a = Array2::<f64>::eye(100);
    for i in 0..100 {
        for j in 0..100 {
            a[[i, j]] = (i as f64 + 1.0) / (j as f64 + 1.0);
        }
    }

    // Ensure the matrix is not singular
    for i in 0..100 {
        a[[i, i]] += 100.0;
    }

    let b = Array1::<f64>::ones(100);

    b.iter(|| test::black_box(a.solve(&b).unwrap()));
}

/// Benchmark solving a linear system with faer
#[bench]
fn bench_faer_solve_100x100(b: &mut Bencher) {
    // Create a matrix and vector
    let mut a = Mat::<f64>::zeros(100, 100);
    for i in 0..100 {
        for j in 0..100 {
            a.write(i, j, (i as f64 + 1.0) / (j as f64 + 1.0));
        }
    }

    // Ensure the matrix is not singular
    for i in 0..100 {
        a.write(i, i, a.read(i, i) + 100.0);
    }

    let b = Mat::<f64>::ones(100, 1);

    b.iter(|| {
        let mut x = Mat::<f64>::zeros(100, 1);
        test::black_box(faer::linalg::solvers::dense::general::solve(
            a.as_ref(),
            x.as_mut(),
            b.as_ref(),
            faer::Parallelism::None,
        ));
    });
}

/// Benchmark solving a linear system with nalgebra
#[bench]
fn bench_nalgebra_solve_100x100(b: &mut Bencher) {
    // Create a matrix and vector
    let mut a = DMatrix::<f64>::zeros(100, 100);
    for i in 0..100 {
        for j in 0..100 {
            a[(i, j)] = (i as f64 + 1.0) / (j as f64 + 1.0);
        }
    }

    // Ensure the matrix is not singular
    for i in 0..100 {
        a[(i, i)] += 100.0;
    }

    let b = DVector::<f64>::repeat(100, 1.0);

    b.iter(|| test::black_box(a.clone().lu().solve(&b).unwrap()));
}
