use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;
use storm::{
    dynamic_interface::{DifferenceSchemes, ErasedSolver},
    model::polytrope::Polytrope,
    system::adiabatic::Rotating1D,
};

fn polytrope(scheme: DifferenceSchemes) -> ErasedSolver {
    let poly = Polytrope::new(3., 5. / 3., 0.01);

    ErasedSolver::new(&poly, Rotating1D::new(0, 0), scheme)
}

fn run_freq(shooting: ErasedSolver) -> f64 {
    shooting.det(10.)
}

fn run_upper(shooting: ErasedSolver) -> Vec<f64> {
    shooting.eigenvector(10.)
}

#[library_benchmark]
#[bench::colloc2(args = (DifferenceSchemes::Colloc2), setup=polytrope)]
#[bench::colloc4(args = (DifferenceSchemes::Colloc4), setup=polytrope)]
#[bench::magnus2(args = (DifferenceSchemes::Magnus2), setup=polytrope)]
#[bench::magnus4(args = (DifferenceSchemes::Magnus4), setup=polytrope)]
#[bench::magnus6(args = (DifferenceSchemes::Magnus6), setup=polytrope)]
#[bench::magnus8(args = (DifferenceSchemes::Magnus8), setup=polytrope)]
fn bench_polytrope(shooting: ErasedSolver) -> f64 {
    black_box(run_freq(shooting))
}

#[library_benchmark]
#[bench::colloc2(args = (DifferenceSchemes::Colloc2), setup=polytrope)]
#[bench::colloc4(args = (DifferenceSchemes::Colloc4), setup=polytrope)]
#[bench::magnus2(args = (DifferenceSchemes::Magnus2), setup=polytrope)]
#[bench::magnus4(args = (DifferenceSchemes::Magnus4), setup=polytrope)]
#[bench::magnus6(args = (DifferenceSchemes::Magnus6), setup=polytrope)]
#[bench::magnus8(args = (DifferenceSchemes::Magnus8), setup=polytrope)]
fn bench_polytrope_eigen(shooting: ErasedSolver) -> Vec<f64> {
    black_box(run_upper(shooting))
}

library_benchmark_group!(
    name = bench_polytrop_group;
    benchmarks = bench_polytrope, bench_polytrope_eigen
);

main!(library_benchmark_groups = bench_polytrop_group);
