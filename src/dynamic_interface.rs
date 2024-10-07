//! Main interface for computing the determinant for a trial frequency

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::solver::{determinant, determinant_with_upper, UpperResult};
use crate::stepper::{Colloc2, Colloc4, Magnus2, Magnus4, Magnus6, Magnus8, Stepper};
use crate::system::System;

/// Supported difference schemes
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum DifferenceSchemes {
    /// Second-order collocation method
    Colloc2,
    /// Fourth-order collocation method
    Colloc4,
    /// Second-order magnus method
    Magnus2,
    /// Fourth-order magnus method
    Magnus4,
    /// Sixt-order magnus method
    Magnus6,
    /// Eigth-order magnus method
    Magnus8,
}

/// Type erased interface for computing the determinant and the eigenvector of a problem
pub struct MultipleShooting<'system_and_grid> {
    det: Box<dyn Fn(f64) -> f64 + 'system_and_grid>,
    eigenvector: Box<dyn Fn(f64) -> (f64, Vec<f64>) + 'system_and_grid>,
}

impl<'system_and_grid> MultipleShooting<'system_and_grid> {
    /// Construct from a system, difference scheme and grid definition
    pub fn new<const N: usize, const N_INNER: usize, G: ?Sized, S>(
        system: &'system_and_grid S,
        scheme: DifferenceSchemes,
        grid: &'system_and_grid G,
    ) -> MultipleShooting<'system_and_grid>
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

    /// Compute the determinant for a certain frequency
    pub fn det(&self, freq: f64) -> f64 {
        (self.det)(freq)
    }

    /// Compute the eigenvectors for a certain frequency.
    ///
    /// This assumes that freq are close to a solution. This is less efficient than
    /// [MultipleShooting::det], so only use this after bracketing has completed.
    pub fn eigenvector(&self, freq: f64) -> Vec<f64> {
        (self.eigenvector)(freq).1
    }

    /// Scan all points given by `freq_grid` and optimize the resulting brackets to `precision`
    pub fn scan_and_optimize<'a>(
        &'a self,
        freq_grid: impl IntoIterator<Item = f64> + 'a,
        precision: Precision,
    ) -> impl Iterator<Item = BracketResult> + 'a {
        freq_grid
            .into_iter()
            .map(|x| Point { x, f: self.det(x) })
            .filter_sign_swap()
            .map(move |(point1, point2)| {
                (InverseQuadratic {})
                    .optimize(
                        point1,
                        point2,
                        |point| Ok::<_, !>(self.det(point)),
                        precision,
                        None,
                    )
                    .into_ok()
            })
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
) -> MultipleShooting<'a>
where
    S: System<f64, G, N, N_INNER, ORDER>,
    [(); { N - N_INNER } * N]:,
    [(); N + N_INNER]:,
    [(); 2 * N]:,
{
    let stepper1 = stepper();
    let stepper2 = stepper();
    MultipleShooting {
        det: Box::new(move |freq: f64| determinant(system, &stepper1, grid, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new(system, grid);

            let det = determinant_with_upper(system, &stepper2, grid, freq, &mut upper);

            (det, upper.eigenvectors())
        }),
    }
}
