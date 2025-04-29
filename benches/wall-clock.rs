use storm::{
    dynamic_interface::{DifferenceSchemes, ErasedSolver},
    model::polytrope::Polytrope,
    system::adiabatic::Rotating1D,
};
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};

pub fn polytrope(scheme: DifferenceSchemes) -> ErasedSolver {
    let poly = Polytrope::new(3., 5. / 3., 0.01);

    ErasedSolver::new(&poly, Rotating1D::new(0, 0), scheme, None)
}

pub fn run_freq(shooting: &ErasedSolver) -> f64 {
    shooting.det(10.)
}

pub fn run_upper(shooting: &ErasedSolver) -> Vec<f64> {
    shooting.eigenvector(10.)
}

fn polytrope_benchmark() -> impl IntoBenchmarks {
    [
        benchmark_fn("polytrope_colloc2", |b| {
            let shooting = polytrope(DifferenceSchemes::Colloc2);
            b.iter(move || run_freq(&shooting))
        }),
        benchmark_fn("polytrope_colloc4", |b| {
            let shooting = polytrope(DifferenceSchemes::Colloc4);
            b.iter(move || run_freq(&shooting))
        }),
        benchmark_fn("polytrope_magnus2", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus2);
            b.iter(move || run_freq(&shooting))
        }),
        benchmark_fn("polytrope_magnus4", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus4);
            b.iter(move || run_freq(&shooting))
        }),
        benchmark_fn("polytrope_magnus6", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus6);
            b.iter(move || run_freq(&shooting))
        }),
        benchmark_fn("polytrope_magnus8", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus8);
            b.iter(move || run_freq(&shooting))
        }),
    ]
}

fn polytrope_upper_benchmark() -> impl IntoBenchmarks {
    [
        benchmark_fn("polytrope_upper_colloc2", |b| {
            let shooting = polytrope(DifferenceSchemes::Colloc2);
            b.iter(move || run_upper(&shooting))
        }),
        benchmark_fn("polytrope_upper_colloc4", |b| {
            let shooting = polytrope(DifferenceSchemes::Colloc4);
            b.iter(move || run_upper(&shooting))
        }),
        benchmark_fn("polytrope_upper_magnus2", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus2);
            b.iter(move || run_upper(&shooting))
        }),
        benchmark_fn("polytrope_upper_magnus4", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus4);
            b.iter(move || run_upper(&shooting))
        }),
        benchmark_fn("polytrope_upper_magnus6", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus6);
            b.iter(move || run_upper(&shooting))
        }),
        benchmark_fn("polytrope_upper_magnus8", |b| {
            let shooting = polytrope(DifferenceSchemes::Magnus8);
            b.iter(move || run_upper(&shooting))
        }),
    ]
}

tango_benchmarks!(polytrope_benchmark(), polytrope_upper_benchmark());
tango_main!();
