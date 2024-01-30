use num::Float;

use crate::linalg::{commutator, Matmul, Matrix};

pub(crate) struct StepMoments<T, const N: usize, const ORDER: usize>
where
    [(); N * N]: Sized,
{
    pub delta: f64,
    pub moments: [Matrix<T, N, N>; ORDER],
}

pub(crate) struct Step<T, const N: usize>
where
    [(); N * N]: Sized,
{
    pub left: Matrix<T, N, N>,
    pub right: Matrix<T, N, N>,
}

pub(crate) trait Stepper<T: Float, const N: usize, const ORDER: usize>
where
    [(); N * N]: Sized,
{
    fn step(&self, step_input: StepMoments<T, N, ORDER>) -> Step<T, N>;
}

pub(crate) struct Magnus2 {}

impl<const N: usize> Stepper<f64, N, 1> for Magnus2
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 1>) -> Step<f64, N> {
        let [mut omega] = step_input.moments;

        omega.exp(step_input.delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus4 {}

impl<const N: usize> Stepper<f64, N, 2> for Magnus4
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 2>) -> Step<f64, N> {
        let [b1, b2] = step_input.moments;
        let delta = step_input.delta;

        let mut omega = b1 - commutator(b1, b2) * (1.0 / 12.0) * delta;

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus6 {}

impl<const N: usize> Stepper<f64, N, 3> for Magnus6
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 3>) -> Step<f64, N> {
        let [b1, b2, b3] = step_input.moments;
        let delta = step_input.delta;

        let c1 = commutator(b1, b2) * delta; // delta
        let c2 = commutator(b1, b3 * 2. + c1) * (-1.0 / 60.) * delta; // delta

        let mut omega =
            b1 + b3 * (1. / 12.) + commutator(b1 * (-20.) - b3 + c1, b2 + c2) * (delta / 240.); // delta

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus8 {}

impl<const N: usize> Stepper<f64, N, 4> for Magnus8
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 4>) -> Step<f64, N> {
        let [b1, b2, b3, b4] = step_input.moments;
        let delta = step_input.delta;

        let s1 = commutator(b1 + b3 * (1. / 28.), b2 + b4 * (3. / 28.)) * (-1. / 28.) * delta;
        let r1 = commutator(b1, b3 * (-1. / 14.) + s1) * (1. / 3.) * delta;
        let s2 = commutator(b1 + b3 * (1. / 28.) + s1, b2 + b4 * (3. / 28.) + r1) * delta;
        let s2_prime = commutator(b2, s1) * delta;
        let r2 = commutator(b1 + s1 * (5. / 4.), b3 * 2.0 + s2 + s2_prime * 0.5) * delta;
        let s3 = commutator(
            b1 + b3 * (1. / 12.) + s1 * (-7. / 3.) + s2 * (-1. / 6.),
            b2 * (-9.) + b4 * (-9. / 4.) + r1 * 63. + r2,
        ) * delta; // delta

        let mut omega = b1 + b3 * (1. / 12.) + s2 * (-7. / 120.) + s3 * (1. / 360.); // delta

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Colloc2 {}

impl<const N: usize> Stepper<f64, N, 1> for Colloc2
where
    [(); N * N]: Sized,
    [(); N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 1>) -> Step<f64, N> {
        let [b1] = step_input.moments;

        let c1 = b1 * step_input.delta * 0.5;
        let c2 = Matrix::eye();

        Step {
            left: c1 + c2,
            right: c1 - c2,
        }
    }
}

pub(crate) struct Colloc4 {}

impl<const N: usize> Stepper<f64, N, 2> for Colloc4
where
    [(); N * N]: Sized,
    [(); N]: Sized,
{
    fn step(&self, step_input: StepMoments<f64, N, 2>) -> Step<f64, N> {
        let [b1, b2] = step_input.moments;
        let delta = step_input.delta;

        let b1 = b1;
        let b2 = b2 * (1. / 12.);

        let inv_mat = b1.matmul((Matrix::eye() + b2 * delta).inv().unwrap()) * delta;

        let c1 = (b1 - inv_mat.matmul(b2)) * (delta * 0.5);
        let c2 = (b2 - inv_mat.matmul(b1) * (1. / 12.)) * delta - Matrix::eye();

        Step {
            left: c1 - c2,
            right: c1 + c2,
        }
    }
}

#[cfg(test)]
mod benches {
    use std::{fmt::Display, fs, path::PathBuf};

    use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration};
    use criterion_macro::criterion;

    use super::{Colloc2, Colloc4, Magnus2, Magnus4, Magnus6, Magnus8};
    use crate::{
        solver::{bracket_search, Bisection},
        system::stretched_string::IntegratedLinearPiecewiseStretchedString,
    };

    fn write_accuracy_results<P: Display>(
        benchmark_name: &str,
        function_name: &str,
        parameter: P,
        result: f64,
    ) {
        let output_dir: PathBuf = [
            "target",
            "criterion",
            benchmark_name,
            function_name,
            parameter.to_string().as_str(),
        ]
        .iter()
        .collect();

        fs::create_dir_all(&output_dir).unwrap();

        fs::write(output_dir.join("result"), format!("{result}")).unwrap();
    }

    fn stretched_string_piecewise(
        lower: f64,
        upper: f64,
        root: f64,
        benchmark_name: &str,
        c: &mut Criterion,
    ) {
        let system = IntegratedLinearPiecewiseStretchedString {};
        let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
        let searcher = Bisection {
            rel_epsilon: f64::EPSILON,
        };
        let mut group = c.benchmark_group(benchmark_name);
        group.plot_config(plot_config);

        for steps in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800] {
            let grid: Vec<_> = (0..steps + 1)
                .map(|n| 1. / steps as f64 * n as f64)
                .collect();

            group.bench_function(BenchmarkId::new("colloc2", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Colloc2 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("colloc4", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Colloc4 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("magnus2", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Magnus2 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("magnus4", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Magnus4 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("magnus6", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Magnus6 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });
            group.bench_function(BenchmarkId::new("magnus8", steps), |b| {
                b.iter(|| {
                    bracket_search(
                        &system,
                        &Magnus8 {},
                        grid.as_slice(),
                        lower,
                        upper,
                        &searcher,
                    )
                    .unwrap()
                })
            });

            write_accuracy_results(
                benchmark_name,
                "colloc2",
                steps,
                bracket_search(
                    &system,
                    &Colloc2 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
            write_accuracy_results(
                benchmark_name,
                "colloc4",
                steps,
                bracket_search(
                    &system,
                    &Colloc4 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
            write_accuracy_results(
                benchmark_name,
                "magnus2",
                steps,
                bracket_search(
                    &system,
                    &Magnus2 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
            write_accuracy_results(
                benchmark_name,
                "magnus4",
                steps,
                bracket_search(
                    &system,
                    &Magnus4 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
            write_accuracy_results(
                benchmark_name,
                "magnus6",
                steps,
                bracket_search(
                    &system,
                    &Magnus6 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
            write_accuracy_results(
                benchmark_name,
                "magnus8",
                steps,
                bracket_search(
                    &system,
                    &Magnus8 {},
                    grid.as_slice(),
                    lower,
                    upper,
                    &searcher,
                )
                .unwrap()
                    - root,
            );
        }

        group.finish();
    }

    #[criterion(Criterion::default().sample_size(10))]
    fn bench_stretched_string_piecewise_small(c: &mut Criterion) {
        const LOWER: f64 = 4.38;
        const UPPER: f64 = 4.376;
        const ROOT: f64 = 4.378157413652409;
        const BENCHMARK_NAME: &str = "piecewise_linear_small_freq";

        stretched_string_piecewise(LOWER, UPPER, ROOT, BENCHMARK_NAME, c);
    }

    #[criterion(Criterion::default().sample_size(10))]
    fn bench_stretched_string_piecewise_medium(c: &mut Criterion) {
        const LOWER: f64 = 107.;
        const UPPER: f64 = 110.;
        const ROOT: f64 = 108.40719198544681;
        const BENCHMARK_NAME: &str = "piecewise_linear_medium_freq";

        stretched_string_piecewise(LOWER, UPPER, ROOT, BENCHMARK_NAME, c);
    }

    #[criterion(Criterion::default().sample_size(10))]
    fn bench_stretched_string_piecewise_large(c: &mut Criterion) {
        const LOWER: f64 = 156.5;
        const UPPER: f64 = 161.;
        const ROOT: f64 = 160.4412762804863;
        const BENCHMARK_NAME: &str = "piecewise_linear_large_freq";

        stretched_string_piecewise(LOWER, UPPER, ROOT, BENCHMARK_NAME, c);
    }
}
