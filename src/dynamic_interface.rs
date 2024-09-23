use crate::solver::{determinant, determinant_with_upper, UpperResult};
use crate::stepper::{Colloc2, Colloc4, Magnus2, Magnus4, Magnus6, Magnus8, Stepper};
use crate::system::System;

#[derive(clap::ValueEnum, Clone, Copy, Default)]
pub enum DifferenceSchemes {
    #[default]
    Colloc2,
    Colloc4,
    Magnus2,
    Magnus4,
    Magnus6,
    Magnus8,
}

pub struct Determinant<'system_and_grid> {
    det: Box<dyn Fn(f64) -> f64 + 'system_and_grid>,
    eigenvector: Box<dyn Fn(f64) -> Vec<f64> + 'system_and_grid>,
}

impl<'system_and_grid> Determinant<'system_and_grid> {
    pub fn det(&self, freq: f64) -> f64 {
        (self.det)(freq)
    }

    pub fn eigenvector(&self, freq: f64) -> Vec<f64> {
        (self.eigenvector)(freq)
    }
}

fn get_solvers_inner<
    'a,
    const N: usize,
    const N_INNER: usize,
    const ORDER: usize,
    G: ?Sized,
    S,
    T: Stepper<f64, N, ORDER> + 'static,
>(
    system: &'a S,
    grid: &'a G,
    stepper: impl Fn() -> T,
) -> Determinant<'a>
where
    S: System<f64, G, N, N_INNER, ORDER>,
    [(); { N - N_INNER } * N]:,
    [(); N + N_INNER]:,
    [(); 2 * N]:,
{
    let stepper1 = stepper();
    let stepper2 = stepper();
    Determinant {
        det: Box::new(move |freq: f64| determinant(system, &stepper1, grid, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new(system, grid);

            let _ = determinant_with_upper(system, &stepper2, grid, freq, &mut upper);

            upper.eigenvectors()
        }),
    }
}

pub fn get_solvers<'a, const N: usize, const N_INNER: usize, G: ?Sized, S>(
    system: &'a S,
    scheme: DifferenceSchemes,
    grid: &'a G,
) -> Determinant<'a>
where
    S: System<f64, G, N, N_INNER, 1>,
    S: System<f64, G, N, N_INNER, 2>,
    S: System<f64, G, N, N_INNER, 3>,
    S: System<f64, G, N, N_INNER, 4>,
    [(); { N - N_INNER } * N]:,
    [(); N + N_INNER]:,
    [(); 2 * N]:,
{
    match scheme {
        DifferenceSchemes::Colloc2 => get_solvers_inner(system, grid, || Colloc2 {}),
        DifferenceSchemes::Colloc4 => get_solvers_inner(system, grid, || Colloc4 {}),
        DifferenceSchemes::Magnus2 => get_solvers_inner(system, grid, || Magnus2 {}),
        DifferenceSchemes::Magnus4 => get_solvers_inner(system, grid, || Magnus4 {}),
        DifferenceSchemes::Magnus6 => get_solvers_inner(system, grid, || Magnus6 {}),
        DifferenceSchemes::Magnus8 => get_solvers_inner(system, grid, || Magnus8 {}),
    }
}
