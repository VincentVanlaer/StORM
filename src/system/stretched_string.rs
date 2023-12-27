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

fn linear_zeroth_moment(start: f64, end: f64, intercept: f64, slope: f64) -> f64 {
    intercept * (end - start) + slope * 0.5 * (end.powi(2) - start.powi(2))
}

impl Moments<f64, 2, 1> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 1] {
        let first_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_zeroth_moment(lower - 0.5, 0.0, 0.5, -0.1)
                    + linear_zeroth_moment(0.0, upper - 0.5, 0.5, 0.1)
            } else {
                linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, -0.1)
            }
        } else {
            linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, 0.1)
        } / (upper - lower);

        [[[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into()]
    }
}

fn linear_first_moment(start: f64, end: f64, intercept: f64, slope: f64, r: f64) -> f64 {
    intercept * 0.5 * (end.powi(2) - start.powi(2))
        + 1. / 3. * (end.powi(3) - start.powi(3)) * slope
        - r * linear_zeroth_moment(start, end, intercept, slope)
}

impl Moments<f64, 2, 2> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 2] {
        let center = lower + (upper - lower) * 0.5 - 0.5;
        let first_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_zeroth_moment(lower - 0.5, 0.0, 0.5, -0.1)
                    + linear_zeroth_moment(0.0, upper - 0.5, 0.5, 0.1)
            } else {
                linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, -0.1)
            }
        } else {
            linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, 0.1)
        } / (upper - lower);

        let second_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_first_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_first_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_first_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_first_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(2);

        [
            [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into(),
            [[0.0, -frequency.powi(2) * second_order * 12.], [0.0, 0.0]].into(),
        ]
    }
}

fn linear_second_moment(start: f64, end: f64, intercept: f64, slope: f64, r: f64) -> f64 {
    // (t - tc)^2 A(t) = (t^2 - 2t * tc + tc^2) A(t) = (t^2 - 2t * tc + tc^2) (intercept + slope t)
    // = (t^2 - 2(t - tc) * tc - tc^2) (intercept + slope t)

    intercept * 1. / 3. * (end.powi(3) - start.powi(3))
        + slope * 0.25 * (end.powi(4) - start.powi(4))
        - 2. * r * linear_first_moment(start, end, intercept, slope, r)
        - r.powi(2) * linear_zeroth_moment(start, end, intercept, slope)
}

impl Moments<f64, 2, 3> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 3] {
        let center = lower + (upper - lower) * 0.5 - 0.5;

        let first_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_zeroth_moment(lower - 0.5, 0.0, 0.5, -0.1)
                    + linear_zeroth_moment(0.0, upper - 0.5, 0.5, 0.1)
            } else {
                linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, -0.1)
            }
        } else {
            linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, 0.1)
        } / (upper - lower);

        let second_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_first_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_first_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_first_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_first_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(2);

        let third_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_second_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_second_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_second_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_second_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(3);

        let third_order_c = linear_second_moment(lower - 0.5, upper - 0.5, 1.0, 0.0, center)
            / (upper - lower).powi(3);

        let b0: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into();
        let b1: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * second_order], [0.0, 0.0]].into();
        let b2: Matrix<f64, 2, 2> = [
            [0.0, -frequency.powi(2) * third_order],
            [third_order_c, 0.0],
        ]
        .into();

        [b0 * (9. / 4.) - b2 * 15., b1 * 12., b0 * (-15.) + b2 * 180.]
    }
}

fn linear_third_moment(start: f64, end: f64, intercept: f64, slope: f64, r: f64) -> f64 {
    // (t - tc)^3 A(t) = (t^3 - 3t^2 * tc + 3t * tc^2 - tc^3) A(t) = (t^3 - 3t^2 * tc + 3t * tc^2 - tc^3) (intercept + slope t)
    // = (t^3 - 3tc * (t^2 - 2t * tc + tc^2) - 3 tc^2 (t - tc) - tc^3) (intercept + slope t)

    intercept * 0.25 * (end.powi(4) - start.powi(4)) + slope * 0.2 * (end.powi(5) - start.powi(5))
        - 3. * r * linear_second_moment(start, end, intercept, slope, r)
        - 3. * r.powi(2) * linear_first_moment(start, end, intercept, slope, r)
        - r.powi(3) * linear_zeroth_moment(start, end, intercept, slope)
}

impl Moments<f64, 2, 4> for IntegratedLinearPiecewiseStretchedString {
    fn evaluate_moments(&self, lower: f64, upper: f64, frequency: f64) -> [Matrix<f64, 2, 2>; 4] {
        let center = lower + (upper - lower) * 0.5 - 0.5;

        let first_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_zeroth_moment(lower - 0.5, 0.0, 0.5, -0.1)
                    + linear_zeroth_moment(0.0, upper - 0.5, 0.5, 0.1)
            } else {
                linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, -0.1)
            }
        } else {
            linear_zeroth_moment(lower - 0.5, upper - 0.5, 0.5, 0.1)
        } / (upper - lower);

        let second_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_first_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_first_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_first_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_first_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(2);

        let third_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_second_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_second_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_second_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_second_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(3);

        let third_order_c = linear_second_moment(lower - 0.5, upper - 0.5, 1.0, 0.0, center)
            / (upper - lower).powi(3);

        let fourth_order = if lower < 0.5 {
            if upper > 0.5 {
                linear_third_moment(lower - 0.5, 0.0, 0.5, -0.1, center)
                    + linear_third_moment(0.0, upper - 0.5, 0.5, 0.1, center)
            } else {
                linear_third_moment(lower - 0.5, upper - 0.5, 0.5, -0.1, center)
            }
        } else {
            linear_third_moment(lower - 0.5, upper - 0.5, 0.5, 0.1, center)
        } / (upper - lower).powi(4);

        let fourth_order_c = linear_third_moment(lower - 0.5, upper - 0.5, 1.0, 0.0, center)
            / (upper - lower).powi(4);

        let b0: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * first_order], [1.0, 0.0]].into();
        let b1: Matrix<f64, 2, 2> = [[0.0, -frequency.powi(2) * second_order], [0.0, 0.0]].into();
        let b2: Matrix<f64, 2, 2> = [
            [0.0, -frequency.powi(2) * third_order],
            [third_order_c, 0.0],
        ]
        .into();
        let b3: Matrix<f64, 2, 2> = [
            [0.0, -frequency.powi(2) * fourth_order],
            [fourth_order_c, 0.0],
        ]
        .into();

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
