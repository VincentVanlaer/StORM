use num::Float;

use crate::{
    jacobian::Interpolator,
    linalg::{Matmul, Matrix},
};

pub(crate) trait Step<T: Float, const N: usize, I: Interpolator<T, N>>
where
    [(); N * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: T, x2: T, frequency: f64) -> Matrix<T, N, N>;
}

pub(crate) struct MagnusGL2 {}

impl<const N: usize, I: Interpolator<f64, N>> Step<f64, N, I> for MagnusGL2
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Matrix<f64, N, N> {
        let mut omega = interpolator.evaluate(0.5 * (x2 + x1), frequency);

        omega.exp(x2 - x1);

        omega
    }
}

pub(crate) struct MagnusGL4 {}

impl<const N: usize, I: Interpolator<f64, N>> Step<f64, N, I> for MagnusGL4
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Matrix<f64, N, N> {
        const SQRT_3: f64 = 1.732_050_807_568_877_2;

        let delta = x2 - x1;
        let a1 = interpolator.evaluate(0.5 * ((x2 + x1) - delta / SQRT_3), frequency);
        let a2 = interpolator.evaluate(0.5 * ((x2 + x1) + delta / SQRT_3), frequency);

        let a41 = (a1 + a2) * 0.5;
        let a42 = (a1 - a2) * SQRT_3;

        let mut omega = a41 - (a41.matmul(a42) - a42.matmul(a41)) * (delta / 12.0);

        omega.exp(delta);

        omega
    }
}

pub(crate) struct MagnusGL6 {}

impl<const N: usize, I: Interpolator<f64, N>> Step<f64, N, I> for MagnusGL6
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Matrix<f64, N, N> {
        const SQRT_5: f64 = 2.236_067_977_499_79;
        const SQRT_3: f64 = 1.732_050_807_568_877_2;

        let delta = x2 - x1;
        let a1 = interpolator.evaluate(0.5 * ((x2 + x1) - delta * SQRT_3 / SQRT_5), frequency);
        let a2 = interpolator.evaluate(0.5 * (x2 + x1), frequency);
        let a3 = interpolator.evaluate(0.5 * ((x2 + x1) + delta * SQRT_3 / SQRT_5), frequency);

        let a61 = a2;
        let a62 = (a3 - a1) * (SQRT_5 / SQRT_3);
        let a63 = (a3 - a2 * 2.0 + a1) * (10. / 3.);

        let c1 = (a61.matmul(a62) - a62.matmul(a61)) * delta;
        let c2_r = a63 * 2.0 + c1;
        let c2 = (a61.matmul(c2_r) - c2_r.matmul(a61)) * (-delta / 60.);

        let fl = a61 * (-20.0) - a63 + c1;
        let fr = a62 + c2;

        let mut omega = a61 + a63 * (1.0 / 12.0) + (fl.matmul(fr) - fr.matmul(fl)) * (delta / 240.);

        omega.exp(delta);

        omega
    }
}
