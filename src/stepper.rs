use lapack::{dgetrf, dgetri};
use num::Float;

use crate::{
    jacobian::Interpolator,
    linalg::{Matrix, commutator},
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
        let a42 = (a2 - a1) * SQRT_3 * delta;

        let mut omega = a41 - commutator(a41, a42) * (1.0 / 12.0);

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

        let a61 = a2;  // delta
        let a62 = (a3 - a1) * (SQRT_5 / SQRT_3) * delta;
        let a63 = (a3 - a2 * 2.0 + a1) * (10. / 3.); // delta

        let c1 = commutator(a61, a62); // delta
        let c2 = commutator(a61, a63 * 2. + c1) * (- 1.0 / 60.) * delta * delta;

        let mut omega = a61 + a63 * (1. / 12.) + commutator(a61 * (-20.) - a63 + c1, a62 + c2) * (1. / 240.); // delta

        omega.exp(delta);

        omega
    }
}

pub(crate) struct MagnusGL8 {
    coefficients: Matrix<f64, 4, 4>
}

const POS1: f64 = 0.33998104358485626480266575910324;
const POS2: f64 = 0.86113631159405257522394648889280;

impl MagnusGL8 {
    pub(crate) fn new() -> MagnusGL8 {
        let mut coefficients: Matrix<f64, 4, 4> = [[0.0; 4]; 4].into();
        let mut ipiv = [0; 4];
        let mut info: i32 = 0;
        let mut work = [0.0; 16];

        for i in 0..4 {
            coefficients[i][0] = (-0.5 * POS2).powi(i as i32);
            coefficients[i][1] = (-0.5 * POS1).powi(i as i32);
            coefficients[i][2] =  (0.5 * POS1).powi(i as i32);
            coefficients[i][3] =  (0.5 * POS2).powi(i as i32);
        }

        unsafe { dgetrf(4, 4, coefficients.as_slice_mut(), 4, &mut ipiv, &mut info) };
        unsafe { dgetri(4, coefficients.as_slice_mut(), 4, &ipiv, &mut work, 16, &mut info) };

        MagnusGL8 { coefficients }
    }
}

impl<const N: usize, I: Interpolator<f64, N>> Step<f64, N, I> for MagnusGL8
where
    [(); N * N]: Sized,
    [(); 4 * N]: Sized,
{
    fn step(&self, interpolator: &I, x1: f64, x2: f64, frequency: f64) -> Matrix<f64, N, N> {

        let delta = x2 - x1;
        let a1 = interpolator.evaluate(0.5 * (x2 + x1 - delta * POS2), frequency);
        let a2 = interpolator.evaluate(0.5 * (x2 + x1 - delta * POS1), frequency);
        let a3 = interpolator.evaluate(0.5 * (x2 + x1 + delta * POS1), frequency);
        let a4 = interpolator.evaluate(0.5 * (x2 + x1 + delta * POS2), frequency);

        let b1 = (a1 * self.coefficients[0][0] + a2 * self.coefficients[1][0] + a3 * self.coefficients[2][0] + a4 * self.coefficients[3][0]); // delta
        let b2 = (a1 * self.coefficients[0][1] + a2 * self.coefficients[1][1] + a3 * self.coefficients[2][1] + a4 * self.coefficients[3][1]) * delta;
        let b3 = (a1 * self.coefficients[0][2] + a2 * self.coefficients[1][2] + a3 * self.coefficients[2][2] + a4 * self.coefficients[3][2]); // delta
        let b4 = (a1 * self.coefficients[0][3] + a2 * self.coefficients[1][3] + a3 * self.coefficients[2][3] + a4 * self.coefficients[3][3]) * delta;

        let s1 = commutator(b1 + b3 * (1. / 28.), b2 + b4 * (3. / 28.)) * (-1. / 28.); // delta
        let r1 = commutator(b1, b3 * (-1. / 14.) + s1) * (1. / 3.) * delta * delta;
        let s2 = commutator(b1 + b3 * (1. / 28.) + s1, b2 + b4 * (3. / 28.) + r1); // delta
        let s2_prime = commutator(b2, s1); // delta
        let r2 = commutator(b1 + s1 * (5./4.), b3 * 2.0 + s2 + s2_prime * 0.5) * delta * delta;
        let s3 = commutator(b1 + b3 * (1. / 12.) + s1 * (-7. / 3.) + s2 * (-1. / 6.), b2 * (-9.) + b4 * (-9. / 4.) + r1 * 63. + r2); // delta

        let mut omega = b1 + b3 * (1. / 12.) + s2 * (-7. / 120.) + s3 * (1. / 360.);  // delta
        
        omega.exp(delta);

        omega
    }
}
