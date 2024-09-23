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

pub mod adiabatic;
