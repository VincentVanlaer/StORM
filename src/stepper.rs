use std::ops::Mul;

use num::Float;

use crate::{
    linalg::{commutator, Matmul, Matrix},
    system::{Moments, PointwiseInterpolator},
};

impl<T: Float + Mul<f64, Output = T>, const N: usize, U: PointwiseInterpolator<T, N>>
    Moments<T, N, 1> for U
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        lower_location: T,
        upper_location: T,
        frequency: T,
    ) -> [Matrix<T, N, N>; 1] {
        [self.evaluate(
            lower_location + (upper_location - lower_location) * 0.5,
            frequency,
        )]
    }
}

impl<
        T: Float + Mul<f64, Output = T> + std::fmt::Display,
        const N: usize,
        U: PointwiseInterpolator<T, N>,
    > Moments<T, N, 2> for U
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        lower_location: T,
        upper_location: T,
        frequency: T,
    ) -> [Matrix<T, N, N>; 2] {
        const SQRT_3: f64 = 1.732_050_807_568_87;
        let delta = upper_location - lower_location;

        let a1 = self.evaluate(
            ((upper_location + lower_location) - delta * (1.0 / SQRT_3)) * 0.5,
            frequency,
        );
        let a2 = self.evaluate(
            ((upper_location + lower_location) + delta * (1.0 / SQRT_3)) * 0.5,
            frequency,
        );

        let b1 = (a1 + a2) * 0.5; // delta
        let b2 = (a2 - a1) * SQRT_3; // delta

        [b1, b2]
    }
}

impl<T: Float + Mul<f64, Output = T>, const N: usize, U: PointwiseInterpolator<T, N>>
    Moments<T, N, 3> for U
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        lower_location: T,
        upper_location: T,
        frequency: T,
    ) -> [Matrix<T, N, N>; 3] {
        const SQRT_5: f64 = 2.236_067_977_499_79;
        const SQRT_3: f64 = 1.732_050_807_568_87;

        let delta = upper_location - lower_location;
        let a1 = self.evaluate(
            ((lower_location + upper_location) - delta * (SQRT_3 / SQRT_5)) * 0.5,
            frequency,
        );
        let a2 = self.evaluate((lower_location + upper_location) * 0.5, frequency);
        let a3 = self.evaluate(
            ((lower_location + upper_location) + delta * (SQRT_3 / SQRT_5)) * 0.5,
            frequency,
        );

        let b1 = a2; // delta
        let b2 = (a3 - a1) * (SQRT_5 / SQRT_3); // delta
        let b3 = (a3 - a2 * 2.0 + a1) * (10. / 3.); // delta

        [b1, b2, b3]
    }
}

impl<T: Float + Mul<f64, Output = T>, const N: usize, U: PointwiseInterpolator<T, N>>
    Moments<T, N, 4> for U
where
    [(); N * N]: Sized,
{
    fn evaluate_moments(
        &self,
        lower_location: T,
        upper_location: T,
        frequency: T,
    ) -> [Matrix<T, N, N>; 4] {
        const POS1: f64 = 0.33998104358485626480266575910324;
        const POS2: f64 = 0.86113631159405257522394648889280;

        const COEFF1: f64 = 0.09232659844072877;
        const COEFF2: f64 = 0.5923265984407289;
        const COEFF3: f64 = 0.21442969527047984;
        const COEFF4: f64 = 3.4844683820901863;
        const COEFF5: f64 = 3.19504825211347;
        const COEFF6: f64 = 7.420540068038946;
        const COEFF7: f64 = 18.795449407555054;

        let delta = upper_location - lower_location;
        let a1 = self.evaluate(
            (upper_location + lower_location - delta * POS2) * 0.5,
            frequency,
        );
        let a2 = self.evaluate(
            (upper_location + lower_location - delta * POS1) * 0.5,
            frequency,
        );
        let a3 = self.evaluate(
            (upper_location + lower_location + delta * POS1) * 0.5,
            frequency,
        );
        let a4 = self.evaluate(
            (upper_location + lower_location + delta * POS2) * 0.5,
            frequency,
        );

        let b1 = a1 * (-COEFF1) + a2 * COEFF2 + a3 * COEFF2 + a4 * (-COEFF1); // delta
        let b2 = a1 * COEFF3 + a2 * (-COEFF4) + a3 * COEFF4 + a4 * (-COEFF3); // delta
        let b3 = a1 * COEFF5 + a2 * (-COEFF5) + a3 * (-COEFF5) + a4 * COEFF5; // delta
        let b4 = a1 * (-COEFF6) + a2 * COEFF7 + a3 * (-COEFF7) + a4 * COEFF6; // delta

        [b1, b2, b3, b4]
    }
}

pub(crate) struct Step<T, const N: usize>
where
    [(); N * N]: Sized,
{
    pub left: Matrix<T, N, N>,
    pub right: Matrix<T, N, N>,
}

pub(crate) trait Stepper<T: Float, const N: usize, const ORDER: usize, I: Moments<T, N, ORDER>>
where
    [(); N * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: T, x2: T, frequency: f64) -> Step<T, N>;
}

pub(crate) struct Magnus2 {}

impl<const N: usize, I: Moments<f64, N, 1>> Stepper<f64, N, 1, I> for Magnus2
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let [mut omega] = interpolator.evaluate_moments(x1, x2, frequency);

        omega.exp(x2 - x1);

        Step {
            left: omega,
            right: Matrix::eye(),
        }
    }
}

pub(crate) struct Magnus4 {}

impl<const N: usize, I: Moments<f64, N, 2>> Stepper<f64, N, 2, I> for Magnus4
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let delta = x2 - x1;
        let [b1, b2] = interpolator.evaluate_moments(x1, x2, frequency);

        let mut omega = b1 - commutator(b1, b2) * (1.0 / 12.0) * delta;

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye(),
        }
    }
}

pub(crate) struct Magnus6 {}

impl<const N: usize, I: Moments<f64, N, 3>> Stepper<f64, N, 3, I> for Magnus6
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let delta = x2 - x1;
        let [b1, b2, b3] = interpolator.evaluate_moments(x1, x2, frequency);

        let c1 = commutator(b1, b2) * delta; // delta
        let c2 = commutator(b1, b3 * 2. + c1) * (-1.0 / 60.) * delta; // delta

        let mut omega =
            b1 + b3 * (1. / 12.) + commutator(b1 * (-20.) - b3 + c1, b2 + c2) * (delta / 240.); // delta

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye(),
        }
    }
}

pub(crate) struct Magnus8 {}

impl<const N: usize, I: Moments<f64, N, 4>> Stepper<f64, N, 4, I> for Magnus8
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let delta = x2 - x1;

        let [b1, b2, b3, b4] = interpolator.evaluate_moments(x1, x2, frequency);

        let s1 = commutator(b1 + b3 * (1. / 28.), b2 + b4 * (3. / 28.)) * (-1. / 28.) * delta;
        let r1 = commutator(b1, b3 * (-1. / 14.) + s1) * (1. / 3.) * delta;
        let s2 = commutator(b1 + b3 * (1. / 28.) + s1, b2 + b4 * (3. / 28.) + r1) * delta;
        let s2_prime = commutator(b2, s1) * delta;
        let r2 = commutator(b1 + s1 * (5. / 4.), b3 * 2.0 + s2 + s2_prime * 0.5) * delta;
        let s3 = commutator(
            b1 + b3 * (1. / 12.) + s1 * (-7. / 3.) + s2 * (-1. / 6.),
            b2 * (-9.) + b4 * (-9. / 4.) + r1 * 63. + r2,
        ) * delta; // delta

        let mut omega = b1 + b3 * (1. / 12.) + s2 * (-7. / 120.) + s3 * (1. / 360.); // delta

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye(),
        }
    }
}

pub(crate) struct Colloc2 {}

impl<const N: usize, I: Moments<f64, N, 1>> Stepper<f64, N, 1, I> for Colloc2
where
    [(); N * N]: Sized,
    [(); 1 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let [b1] = interpolator.evaluate_moments(x1, x2, frequency);

        let c1 = b1 * (x2 - x1) * 0.5;
        let c2 = Matrix::eye();

        Step {
            left: c1 + c2,
            right: c1 - c2,
        }
    }
}

pub(crate) struct Colloc4 {}

impl<const N: usize, I: Moments<f64, N, 2>> Stepper<f64, N, 2, I> for Colloc4
where
    [(); N * N]: Sized,
    [(); 1 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Step<f64, N> {
        let [b1, b2] = interpolator.evaluate_moments(x1, x2, frequency);
        let delta = x2 - x1;

        let b1 = b1;
        let b2 = b2 * (1. / 12.);

        let inv_mat = b1.matmul((Matrix::eye() + b2 * delta).inv().unwrap()) * delta;

        let c1 = (b1 - inv_mat.matmul(b2)) * (delta * 0.5);
        let c2 = (b2 - inv_mat.matmul(b1) * (1. / 12.)) * delta - Matrix::eye();

        Step {
            left: c1 - c2,
            right: c1 + c2,
        }
    }
}
