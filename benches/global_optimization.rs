//! Benchmarks for global optimization methods.
//!
//! This benchmark compares the performance of different global optimization
//! algorithms on standard test functions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lmopt_rs::error::{LmOptError, Result};
use lmopt_rs::global_opt::{
    optimize_global, optimize_global_parallel, BasinHopping, DifferentialEvolution,
    GlobalOptimizer, HybridGlobal, ParallelDifferentialEvolution, SimulatedAnnealing,
};
use lmopt_rs::problem::Problem;
use ndarray::{array, Array1};
use std::f64::consts::PI;

// --- Standard Test Functions ---

/// Rastrigin function: f(x) = 10n + sum[x_i^2 - 10cos(2πx_i)]
struct RastriginFunction {
    dimension: usize,
}

impl RastriginFunction {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl Problem for RastriginFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let mut sum = 10.0 * self.dimension as f64;
        for x in params.iter() {
            sum += x.powi(2) - 10.0 * (2.0 * PI * x).cos();
        }

        Ok(array![sum.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

/// Rosenbrock function: f(x) = sum[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
struct RosenbrockFunction {
    dimension: usize,
}

impl RosenbrockFunction {
    fn new(dimension: usize) -> Self {
        if dimension < 2 {
            panic!("Rosenbrock function requires at least 2 dimensions");
        }
        Self { dimension }
    }
}

impl Problem for RosenbrockFunction {
    fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
        if params.len() != self.dimension {
            return Err(LmOptError::DimensionMismatch(format!(
                "Expected {} parameters, got {}",
                self.dimension,
                params.len()
            )));
        }

        let mut sum = 0.0;
        for i in 0..self.dimension - 1 {
            let term1 = 100.0 * (params[i + 1] - params[i].powi(2)).powi(2);
            let term2 = (1.0 - params[i]).powi(2);
            sum += term1 + term2;
        }

        Ok(array![sum.sqrt()])
    }

    fn parameter_count(&self) -> usize {
        self.dimension
    }

    fn residual_count(&self) -> usize {
        1
    }
}

// --- Benchmark Functions ---

fn bench_rastrigin(c: &mut Criterion) {
    let mut group = c.benchmark_group("rastrigin_2d");
    group.sample_size(10); // Reduce sample size for slow benchmarks

    let problem = RastriginFunction::new(2);
    let bounds = vec![(-5.12, 5.12); 2];

    // Benchmark Differential Evolution
    group.bench_function("differential_evolution", |b| {
        b.iter(|| {
            let optimizer = DifferentialEvolution::new()
                .with_population_size(30)
                .with_crossover_probability(0.9)
                .with_differential_weight(0.8);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(20), // Reduced iterations for benchmarking
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    // Benchmark Parallel Differential Evolution
    group.bench_function("parallel_differential_evolution", |b| {
        b.iter(|| {
            let optimizer = ParallelDifferentialEvolution::new()
                .with_population_size(30)
                .with_crossover_probability(0.9)
                .with_differential_weight(0.8);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    // Benchmark Basin Hopping
    group.bench_function("basin_hopping", |b| {
        b.iter(|| {
            let optimizer = BasinHopping::new()
                .with_step_size(0.5)
                .with_local_optimization(true);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    // Benchmark Simulated Annealing
    group.bench_function("simulated_annealing", |b| {
        b.iter(|| {
            let optimizer = SimulatedAnnealing::new()
                .with_initial_temperature(10.0)
                .with_cooling_rate(0.95);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    // Benchmark Hybrid Global
    group.bench_function("hybrid_global", |b| {
        b.iter(|| {
            let optimizer = HybridGlobal::new();

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    // Benchmark convenience functions
    group.bench_function("optimize_global", |b| {
        b.iter(|| {
            let _ = optimize_global(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    group.bench_function("optimize_global_parallel", |b| {
        b.iter(|| {
            let _ = optimize_global_parallel(
                &problem,
                black_box(&bounds),
                black_box(20),
                black_box(10),
                black_box(1e-4),
            );
        })
    });

    group.finish();
}

fn bench_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("rosenbrock");
    group.sample_size(10); // Reduce sample size for slow benchmarks

    // Benchmark with different dimensions
    for dim in [2, 5, 10] {
        let problem = RosenbrockFunction::new(dim);
        let bounds = vec![(-5.0, 10.0); dim];

        // Benchmark Differential Evolution
        group.bench_with_input(
            BenchmarkId::new("differential_evolution", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let optimizer = DifferentialEvolution::new()
                        .with_population_size(30)
                        .with_crossover_probability(0.9)
                        .with_differential_weight(0.8);

                    let _ = optimizer.optimize(
                        &problem,
                        black_box(&bounds),
                        black_box(20),
                        black_box(10),
                        black_box(1e-4),
                    );
                })
            },
        );

        // Benchmark Parallel Differential Evolution
        group.bench_with_input(
            BenchmarkId::new("parallel_differential_evolution", dim),
            &dim,
            |b, _| {
                b.iter(|| {
                    let optimizer = ParallelDifferentialEvolution::new()
                        .with_population_size(30)
                        .with_crossover_probability(0.9)
                        .with_differential_weight(0.8);

                    let _ = optimizer.optimize(
                        &problem,
                        black_box(&bounds),
                        black_box(20),
                        black_box(10),
                        black_box(1e-4),
                    );
                })
            },
        );
    }

    group.finish();
}

fn bench_population_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_size");
    group.sample_size(10); // Reduce sample size for slow benchmarks

    let problem = RastriginFunction::new(5);
    let bounds = vec![(-5.12, 5.12); 5];

    // Benchmark with different population sizes
    for pop_size in [10, 30, 50, 100] {
        // Sequential DE
        group.bench_with_input(
            BenchmarkId::new("differential_evolution", pop_size),
            &pop_size,
            |b, &size| {
                b.iter(|| {
                    let optimizer = DifferentialEvolution::new()
                        .with_population_size(size)
                        .with_crossover_probability(0.9)
                        .with_differential_weight(0.8);

                    let _ = optimizer.optimize(
                        &problem,
                        black_box(&bounds),
                        black_box(20),
                        black_box(10),
                        black_box(1e-4),
                    );
                })
            },
        );

        // Parallel DE
        group.bench_with_input(
            BenchmarkId::new("parallel_differential_evolution", pop_size),
            &pop_size,
            |b, &size| {
                b.iter(|| {
                    let optimizer = ParallelDifferentialEvolution::new()
                        .with_population_size(size)
                        .with_crossover_probability(0.9)
                        .with_differential_weight(0.8);

                    let _ = optimizer.optimize(
                        &problem,
                        black_box(&bounds),
                        black_box(20),
                        black_box(10),
                        black_box(1e-4),
                    );
                })
            },
        );
    }

    group.finish();
}

fn bench_expensive_function(c: &mut Criterion) {
    // Create a problem with expensive function evaluation
    struct ExpensiveFunction {
        dimension: usize,
        iterations: usize, // Controls the computational cost
    }

    impl ExpensiveFunction {
        fn new(dimension: usize, iterations: usize) -> Self {
            Self {
                dimension,
                iterations,
            }
        }
    }

    impl Problem for ExpensiveFunction {
        fn eval(&self, params: &Array1<f64>) -> Result<Array1<f64>> {
            if params.len() != self.dimension {
                return Err(LmOptError::DimensionMismatch(format!(
                    "Expected {} parameters, got {}",
                    self.dimension,
                    params.len()
                )));
            }

            // Simulate expensive computation
            let mut sum = 0.0;
            for _ in 0..self.iterations {
                for i in 0..self.dimension {
                    sum += (params[i].sin() * params[i].cos()).powi(2);
                }
            }

            // Final value is a simple function plus the computation overhead
            let result = params.iter().map(|&x| x.powi(2)).sum::<f64>() + sum * 1e-10;

            Ok(array![result.sqrt()])
        }

        fn parameter_count(&self) -> usize {
            self.dimension
        }

        fn residual_count(&self) -> usize {
            1
        }
    }

    let mut group = c.benchmark_group("expensive_function");
    group.sample_size(10); // Reduce sample size for slow benchmarks

    // Create problem with adjustable computational cost
    let problem = ExpensiveFunction::new(5, 1000); // 5D problem with expensive evaluation
    let bounds = vec![(-5.0, 5.0); 5];

    // Benchmark Sequential DE
    group.bench_function("differential_evolution", |b| {
        b.iter(|| {
            let optimizer = DifferentialEvolution::new()
                .with_population_size(20)
                .with_crossover_probability(0.9)
                .with_differential_weight(0.8);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(10), // Very few iterations for benchmarking
                black_box(5),
                black_box(1e-4),
            );
        })
    });

    // Benchmark Parallel DE
    group.bench_function("parallel_differential_evolution", |b| {
        b.iter(|| {
            let optimizer = ParallelDifferentialEvolution::new()
                .with_population_size(20)
                .with_crossover_probability(0.9)
                .with_differential_weight(0.8);

            let _ = optimizer.optimize(
                &problem,
                black_box(&bounds),
                black_box(10),
                black_box(5),
                black_box(1e-4),
            );
        })
    });

    group.finish();
}

// Combine all benchmarks
criterion_group!(
    benches,
    bench_rastrigin,
    bench_rosenbrock,
    bench_population_size,
    bench_expensive_function
);
criterion_main!(benches);
