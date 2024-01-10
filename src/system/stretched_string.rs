use num::Float;

use super::Boundary;
use super::Moments;
use super::PointwiseInterpolator;
use crate::linalg::Matrix;

pub(crate) struct StretchedString {
    pub speed_generator: fn(f64) -> f64,
}

pub(crate) fn constant_speed(_location: f64) -> f64 {
    1.0
}

pub(crate) fn parabola(location: f64) -> f64 {
    2.0 * (location - 0.5).powi(2) + 0.5
}

pub(crate) fn linear_piecewise(location: f64) -> f64 {
    0.1 * (location - 0.5).abs() + 0.5
}

pub(crate) fn smoothened_linear_piecewise(location: f64) -> f64 {
    let offset = location - 0.5;
    0.5 + 0.1 * (offset.powi(2) + 0.00001).sqrt()
}

impl PointwiseInterpolator<f64, 2> for StretchedString {
    fn evaluate(&self, location: f64, frequency: f64) -> Matrix<f64, 2, 2> {
        [
            [0.0, -frequency.powi(2) * (self.speed_generator)(location)],
            [1.0, 0.0],
        ]
        .into()
    }
}

impl Boundary<f64, 2, 1, 1> for StretchedString {
    fn inner_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }
}

pub(crate) struct IntegratedLinearPiecewiseStretchedString {}

fn linear_zeroth_moment(_s: f64, i: f64, _c: f64) -> f64 {
    i
}

fn linear_first_moment(s: f64, i: f64, c: f64) -> f64 {
    s / 12. - c * i
}

fn linear_second_moment(s: f64, i: f64, c: f64) -> f64 {
    -c / 6. * s + (c.powi(2) + 1. / 12.) * i
}

fn linear_third_moment(s: f64, i: f64, c: f64) -> f64 {
    (c.powi(2) / 4. + 1. / 80.) * s - (-c.powi(3) - c / 4.) * i
}

fn compute_moment(lower: f64, upper: f64, moment: fn(f64, f64, f64) -> f64, n: i32) -> f64 {
    let delta = upper - lower;
    let partial_lower_scale = ((0.5 - lower) / delta).powi(n + 1);
    let partial_upper_scale = ((upper - 0.5) / delta).powi(n + 1);

    if lower < 0.5 {
        if upper > 0.5 {
            let partial_lower = moment(
                -0.1 * delta,
                linear_piecewise(0.5 * (lower + 0.5)),
                0.5 - lower,
            );
            let partial_upper = moment(
                0.1 * delta,
                linear_piecewise(0.5 * (upper + 0.5)),
                0.5 - upper,
            );
            partial_lower * partial_lower_scale + partial_upper * partial_upper_scale
        } else {
            moment(-0.1 * delta, linear_piecewise(0.5 * (lower + upper)), 0.0)
        }
    } else {
        moment(0.1 * delta, linear_piecewise(0.5 * (lower + upper)), 0.0)
    }
}

impl Moments<f64, 2, 1> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 1] {
        let first_order = compute_moment(lower, upper, linear_zeroth_moment, 0);

        [[[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into()]
    }
}

impl Moments<f64, 2, 2> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 2] {
        let first_order = compute_moment(lower, upper, linear_zeroth_moment, 0);
        let second_order = compute_moment(lower, upper, linear_first_moment, 1);

        [
            [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into(),
            [[0.0, -frequency.powi(2) * second_order * 12.], [0.0, 0.0]].into(),
        ]
    }
}

impl Moments<f64, 2, 3> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 3] {
        let first_order = compute_moment(lower, upper, linear_zeroth_moment, 0);
        let second_order = compute_moment(lower, upper, linear_first_moment, 1);
        let third_order = compute_moment(lower, upper, linear_second_moment, 2);

        let b0: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into();
        let b1: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * second_order], [0.0, 0.0]].into();
        let b2: Matrix<f64, 2, 2> =
            [[0.0, -frequency.powi(2) * third_order], [1. / 12., 0.0]].into();

        [b0 * (9. / 4.) - b2 * 15., b1 * 12., b0 * (-15.) + b2 * 180.]
    }
}

impl Moments<f64, 2, 4> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 4] {
        let first_order = compute_moment(lower, upper, linear_zeroth_moment, 0);
        let second_order = compute_moment(lower, upper, linear_first_moment, 1);
        let third_order = compute_moment(lower, upper, linear_second_moment, 2);
        let fourth_order = compute_moment(lower, upper, linear_third_moment, 3);

        let b0: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into();
        let b1: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * second_order], [0.0, 0.0]].into();
        let b2: Matrix<f64, 2, 2> =
            [[0.0, -frequency.powi(2) * third_order], [1. / 12., 0.0]].into();
        let b3: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * fourth_order], [0.0, 0.0]].into();

        [
            b0 * (9. / 4.) - b2 * 15.,
            b1 * 75. - b3 * 420.,
            b0 * (-15.) + b2 * 180.,
            b1 * (-420.) + b3 * 2800.,
        ]
    }
}

impl Boundary<f64, 2, 1, 1> for IntegratedLinearPiecewiseStretchedString {
    fn inner_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }
}
