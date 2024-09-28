//! Systems of equations, interface and implementations
use crate::{linalg::Matrix, stepper::StepMoments};
use num::Float;

pub(crate) trait Moments<T: Float, G: ?Sized, const N: usize, const ORDER: usize> {
    fn evaluate_moments(
        &self,
        grid: &G,
        frequency: T,
    ) -> impl ExactSizeIterator<Item = StepMoments<T, N, ORDER>>;
}

pub(crate) trait GridLength<G: ?Sized> {
    fn len(&self, grid: &G) -> usize;
}

pub(crate) trait Boundary<T: Float, const N: usize, const N_INNER: usize> {
    fn inner_boundary(&self, frequency: f64) -> Matrix<T, N, N_INNER>;
    fn outer_boundary(&self, _frequency: f64) -> Matrix<T, N, { N - N_INNER }>;
}

#[allow(private_bounds)]
/// Represents a system of equations with boundary conditions.
///
/// This trait is still a bit in flux, and hence the parent traits are private, making it not
/// possible for downstream crates to implement or call any methods on this trait. In the future,
/// this trait may become fully public.
///
/// This trait has many generic parameters, to allow for as much optimizations as possible. The
/// generic parameters mean the following:
///
/// - `T`: the base numeric type, e.g. f64
/// - `G`: the gridding method. This needs to be a generic method to allow for full inlining of the
///   main loop
/// - `N`: number of equations per point in the system of equations, used for loop unrolling
/// - `N_INNER`: number of inner boundary conditioms
/// - `ORDER`: order of stepping method. The lower the order, the less information needs to be
///   computed.
pub trait System<T: Float, G: ?Sized, const N: usize, const N_INNER: usize, const ORDER: usize>:
    Moments<T, G, N, ORDER> + Boundary<T, N, N_INNER> + GridLength<G>
{
}

impl<T: Float, G: ?Sized, const N: usize, const N_INNER: usize, const ORDER: usize, U>
    System<T, G, N, N_INNER, ORDER> for U
where
    U: Moments<T, G, N, ORDER> + Boundary<T, N, N_INNER> + GridLength<G>,
{
}

/// Adiabatic stellar oscillation equations
pub mod adiabatic;
