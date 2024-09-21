use color_eyre::eyre::eyre;
use color_eyre::Result;

use crate::solver::{
    calc_n_bands, decompose_system_matrix, direct_determinant, DecomposedSystemMatrix,
};
use crate::stepper::{Colloc2, Colloc4, Magnus2, Magnus4, Magnus6, Magnus8};
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

pub fn get_solvers<'a, const N: usize, const N_INNER: usize, G: ?Sized, S>(
    system: &'a S,
    scheme: DifferenceSchemes,
    grid: &'a G,
) -> (
    Box<dyn Fn(f64) -> Result<DecomposedSystemMatrix> + 'a>,
    Box<dyn Fn(f64) -> f64 + 'a>,
)
where
    S: System<f64, G, N, N_INNER, 1>,
    S: System<f64, G, N, N_INNER, 2>,
    S: System<f64, G, N, N_INNER, 3>,
    S: System<f64, G, N, N_INNER, 4>,
    [(); { N - N_INNER } * N]:,
    [(); N + N_INNER]:,
    [(); 2 * N]:,
    [(); calc_n_bands::<N, N_INNER>()]:,
{
    match scheme {
        DifferenceSchemes::Colloc2 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Colloc2 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Colloc2 {}, grid, freq)),
        ),
        DifferenceSchemes::Colloc4 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Colloc4 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Colloc4 {}, grid, freq)),
        ),
        DifferenceSchemes::Magnus2 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Magnus2 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Magnus2 {}, grid, freq)),
        ),
        DifferenceSchemes::Magnus4 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Magnus4 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Magnus4 {}, grid, freq)),
        ),
        DifferenceSchemes::Magnus6 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Magnus6 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Magnus6 {}, grid, freq)),
        ),
        DifferenceSchemes::Magnus8 => (
            Box::new(|freq: f64| {
                decompose_system_matrix(system, &Magnus8 {}, grid, freq)
                    .or(Err(eyre!("Failed determinant")))
            }),
            Box::new(|freq: f64| direct_determinant(system, &Magnus8 {}, grid, freq)),
        ),
    }
}
