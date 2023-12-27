use crate::linalg::Matrix;
use num::Float;

pub(crate) trait PointwiseInterpolator<T: Float, const N: usize>
where
    [(); N * N]: Sized,
{
    fn evaluate(&self, location: T, frequency: T) -> Matrix<T, N, N>;
}

pub(crate) trait Moments<T: Float, const N: usize, const ORDER: usize>
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        lower_location: T,
        upper_location: T,
        frequency: T,
    ) -> [Matrix<T, N, N>; ORDER];
}

pub(crate) trait Boundary<T: Float, const N: usize, const N_INNER: usize, const N_OUTER: usize>
where
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
{
    fn inner_boundary(&self, frequency: f64) -> Matrix<T, N_INNER, N>;
    fn outer_boundary(&self, _frequency: f64) -> Matrix<T, N_OUTER, N>;
}

pub(crate) trait System<
    T: Float,
    const N: usize,
    const N_INNER: usize,
    const N_OUTER: usize,
    const ORDER: usize,
>: Moments<T, N, ORDER> + Boundary<T, N, N_INNER, N_OUTER> where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
{
}

impl<
        T: Float,
        const N: usize,
        const N_INNER: usize,
        const N_OUTER: usize,
        const ORDER: usize,
        U,
    > System<T, N, N_INNER, N_OUTER, ORDER> for U
where
    U: Moments<T, N, ORDER> + Boundary<T, N, N_INNER, N_OUTER>,
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
{
}

pub(crate) mod adiabatic;
pub(crate) mod stretched_string;
