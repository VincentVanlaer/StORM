use std::{num::NonZeroU64, ops::ControlFlow};

pub struct BracketResult {
    pub freq: f64,
    pub evals: u64,
}

pub trait BracketSearcher {
    type InternalState;

    fn search<E>(
        &self,
        lower: Point,
        upper: Point,
        f: impl Fn(f64) -> Result<f64, E>,
        f_callback: Option<&mut dyn FnMut(Self::InternalState)>,
    ) -> Result<BracketResult, E>;
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub f: f64,
}

pub struct Balanced {
    pub rel_epsilon: f64,
}

fn evaluate_secant(x1: Point, x2: Point, y: f64) -> f64 {
    (y - x2.f) / (x1.f - x2.f) * x1.x + (y - x1.f) / (x2.f - x1.f) * x2.x
}

fn evaluate_inverse_quadratic(x1: Point, x2: Point, x3: Point, y: f64) -> f64 {
    (y - x2.f) * (y - x3.f) / (x1.f - x2.f) / (x1.f - x3.f) * x1.x
        + (y - x1.f) * (y - x3.f) / (x2.f - x1.f) / (x2.f - x3.f) * x2.x
        + (y - x1.f) * (y - x2.f) / (x3.f - x2.f) / (x3.f - x1.f) * x3.x
}

// The purpose of this function is:
// 1. Ensure that we always make progress, e.g. in case the secant method returns either the upper
//    or the lower point due to rounding.
// 2. Ensure that bracketing can terminate as early as possible for a given requested precision.
//    This is accomplished by not allowing x to be closer to lower or upper than the rel_epsilon
//    limit, i.e. (x - closer).abs() <= epsilon. If upper and lower are at most 2 * epsilon close,
//    then (upper - x) <= epsilon and (lower - x) <= epsilon is guaranteed, but where exactly x
//    will fall is not.
// 3. Since all of the funny FP math is in here, it is a good place to check whether the bracket is
//    small enough
//
// This function gets the precision in number of ULPs between upper and lower. This avoids all
// sorts of edge cases.
fn ensure_maximal_bracket(
    upper: f64,
    lower: f64,
    x: f64,
    requested_ulp_precision: NonZeroU64,
) -> ControlFlow<f64, f64> {
    assert!(upper > lower);
    assert!(upper.is_finite());
    assert!(lower.is_finite());
    assert!(upper > 0.);
    assert!(lower > 0.);

    let requested_ulp_precision: u64 = requested_ulp_precision.into();

    // We need to be able to multiply by two, and it doesn't really make sense to have more ULPs
    // than space in the mantissa in the final bracket
    assert!(requested_ulp_precision < 2_u64.pow(54));

    let upper = upper.to_bits();
    let lower = lower.to_bits();

    if upper - lower <= requested_ulp_precision {
        return ControlFlow::Break(x);
    }

    let mut x = x.to_bits();

    if (upper - lower) <= 2 * requested_ulp_precision {
        x = lower + (upper - lower) / 2;
    } else if (upper < x) || (upper - x) < requested_ulp_precision {
        x = upper - requested_ulp_precision;
    } else if (lower > x) || (x - lower) < requested_ulp_precision {
        x = lower + requested_ulp_precision;
    }

    ControlFlow::Continue(f64::from_bits(x))
}

pub struct BalancedState {
    pub upper: Point,
    pub lower: Point,
    pub previous: Option<Point>,
    pub offset: f64,
    pub next_eval: f64,
}

impl BracketSearcher for Balanced {
    type InternalState = BalancedState;

    fn search<E>(
        &self,
        mut lower: Point,
        mut upper: Point,
        f: impl Fn(f64) -> Result<f64, E>,
        mut f_callback: Option<&mut dyn FnMut(Self::InternalState)>,
    ) -> Result<BracketResult, E> {
        if lower.f.signum() == upper.f.signum() {
            panic!("Upper and lower values in bracket have the same sign????");
        }

        let mut evals = 0;
        let mut previous: Option<Point> = None;
        let requested_ulp_precision = NonZeroU64::new(u64::max(
            1,
            (self.rel_epsilon / f64::EPSILON).floor() as u64,
        ))
        .expect("max forces >= 1");

        loop {
            let closer;
            let further;

            if upper.f.abs() < lower.f.abs() {
                closer = upper;
                further = lower;
            } else {
                closer = lower;
                further = upper;
            }

            let lfrac = (further.f.abs() / closer.f.abs()).log10();
            let c3 = 0.0 / (1. + ((4. - lfrac) * 3.).exp());

            let x = match previous {
                None => evaluate_secant(lower, upper, -c3 * closer.f),
                Some(p) => {
                    let mut x = evaluate_inverse_quadratic(lower, upper, p, -c3 * closer.f);

                    if x <= lower.x || x >= upper.x {
                        x = evaluate_secant(lower, upper, -c3 * closer.f);
                    }

                    x
                }
            };

            let x = match ensure_maximal_bracket(upper.x, lower.x, x, requested_ulp_precision) {
                ControlFlow::Break(x) => return Ok(BracketResult { freq: x, evals }),
                ControlFlow::Continue(x) => x,
            };

            if let Some(ref mut cb) = f_callback {
                cb(BalancedState {
                    lower,
                    upper,
                    previous,
                    offset: -c3 * closer.f,
                    next_eval: x,
                });
            }

            let next = Point { x, f: f(x)? };

            evals += 1;

            if upper.f.signum() == next.f.signum() {
                if previous.is_none() || previous.unwrap().f.abs() > upper.f.abs() {
                    previous = Some(upper);
                }

                upper = next;
            } else {
                if previous.is_none() || previous.unwrap().f.abs() > lower.f.abs() {
                    previous = Some(lower);
                }

                lower = next;
            }
        }
    }
}

pub trait SearchBrackets {
    type Out;

    fn brackets(self) -> impl Iterator<Item = Self::Out>;
}

impl<'a, T: Iterator<Item = &'a Point>> SearchBrackets for T {
    type Out = (&'a Point, &'a Point);

    fn brackets(self) -> impl Iterator<Item = (&'a Point, &'a Point)> {
        self.map_windows(|[pair1, pair2]| {
            if pair1.f.signum() != pair2.f.signum() {
                Some((*pair1, *pair2))
            } else {
                None
            }
        })
        .filter_map(|x| x)
    }
}
