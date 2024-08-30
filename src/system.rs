use crate::{linalg::Matrix, stepper::StepMoments};
use num::Float;

pub(crate) trait Moments<T: Float, G: ?Sized, const N: usize, const ORDER: usize>
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        grid: &G,
        frequency: T,
    ) -> impl ExactSizeIterator<Item = StepMoments<T, N, ORDER>>;
}

pub(crate) trait Boundary<T: Float, const N: usize, const N_INNER: usize>
where
    [(); N_INNER * N]: Sized,
    [(); {N - N_INNER} * N]: Sized,
{
    fn inner_boundary(&self, frequency: f64) -> Matrix<T, N_INNER, N>;
    fn outer_boundary(&self, _frequency: f64) -> Matrix<T, {N - N_INNER}, N>;
}

pub(crate) trait System<
    T: Float,
    G: ?Sized,
    const N: usize,
    const N_INNER: usize,
    const ORDER: usize,
>: Moments<T, G, N, ORDER> + Boundary<T, N, N_INNER> where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); {N - N_INNER} * N]: Sized,
{
}

impl<
        T: Float,
        G: ?Sized,
        const N: usize,
        const N_INNER: usize,
        const ORDER: usize,
        U,
    > System<T, G, N, N_INNER, ORDER> for U
where
    U: Moments<T, G, N, ORDER> + Boundary<T, N, N_INNER>,
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); {N - N_INNER} * N]: Sized,
{
}

pub mod adiabatic;
pub(crate) mod stretched_string;
