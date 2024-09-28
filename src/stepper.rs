use num::Float;

use crate::linalg::{commutator, Matmul, Matrix};

pub(crate) struct StepMoments<T, const N: usize, const ORDER: usize> {
    pub delta: f64,
    pub moments: [Matrix<T, N, N>; ORDER],
}

pub(crate) struct Step<T, const N: usize> {
    pub left: Matrix<T, N, N>,
    pub right: Matrix<T, N, N>,
}

pub(crate) trait Stepper<T: Float, const N: usize, const ORDER: usize> {
    fn step(&self, step_input: StepMoments<T, N, ORDER>) -> Step<T, N>;
}

pub(crate) struct Magnus2 {}

impl<const N: usize> Stepper<f64, N, 1> for Magnus2 {
    fn step(&self, step_input: StepMoments<f64, N, 1>) -> Step<f64, N> {
        let [mut omega] = step_input.moments;

        omega.exp(step_input.delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus4 {}

impl<const N: usize> Stepper<f64, N, 2> for Magnus4 {
    fn step(&self, step_input: StepMoments<f64, N, 2>) -> Step<f64, N> {
        let [b1, b2] = step_input.moments;
        let delta = step_input.delta;

        let mut omega = b1 - commutator(b1, b2) * (1.0 / 12.0) * delta;

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus6 {}

impl<const N: usize> Stepper<f64, N, 3> for Magnus6 {
    fn step(&self, step_input: StepMoments<f64, N, 3>) -> Step<f64, N> {
        let [b1, b2, b3] = step_input.moments;
        let delta = step_input.delta;

        let c1 = commutator(b1, b2) * delta; // delta
        let c2 = commutator(b1, b3 * 2. + c1) * (-1.0 / 60.) * delta; // delta

        let mut omega =
            b1 + b3 * (1. / 12.) + commutator(b1 * (-20.) - b3 + c1, b2 + c2) * (delta / 240.); // delta

        omega.exp(delta);

        Step {
            left: omega,
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Magnus8 {}

impl<const N: usize> Stepper<f64, N, 4> for Magnus8 {
    fn step(&self, step_input: StepMoments<f64, N, 4>) -> Step<f64, N> {
        let [b1, b2, b3, b4] = step_input.moments;
        let delta = step_input.delta;

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
            right: Matrix::eye() * (-1.),
        }
    }
}

pub(crate) struct Colloc2 {}

impl<const N: usize> Stepper<f64, N, 1> for Colloc2 {
    #[inline(always)]
    fn step(&self, step_input: StepMoments<f64, N, 1>) -> Step<f64, N> {
        let [b1] = step_input.moments;

        let c1 = b1 * step_input.delta * 0.5;
        let c2 = Matrix::eye();

        Step {
            left: c1 + c2,
            right: c1 - c2,
        }
    }
}

pub(crate) struct Colloc4 {}

impl<const N: usize> Stepper<f64, N, 2> for Colloc4 {
    #[inline(always)]
    fn step(&self, step_input: StepMoments<f64, N, 2>) -> Step<f64, N> {
        let [b1, b2] = step_input.moments;
        let delta = step_input.delta;

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
