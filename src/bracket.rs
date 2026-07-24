//! Root finding using bracketing methods
use itertools::Itertools;
use std::{num::NonZeroU64, ops::ControlFlow};

/// Result of optimizing a root finding bracket
#[derive(Debug, Clone, Copy)]
pub struct BracketResult {
    /// Lower limit of the bracket
    pub lower: Point,
    /// Upper limit of the bracket
    pub upper: Point,
    /// Best estimation of the root (assuming there is only one)
    pub root: f64,
    /// Number of function evaluations used to reach the requested precision
    pub evals: u64,
}

/// Wrapper type for providing precision in different ways
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    /// Units in last place.
    ///
    /// How many consecutive floating points the upper and lower bracket can be at most
    ULP(NonZeroU64),
    /// Relative precision
    ///
    /// Converted to ULPs by `rel / EPSILON`
    Relative(f64),
}

impl Precision {
    fn as_ulp(&self) -> NonZeroU64 {
        match self {
            Precision::ULP(u) => *u,
            Precision::Relative(f) => {
                NonZeroU64::new(u64::max(1, (f / f64::EPSILON).floor() as u64))
                    .expect("max forces >= 1")
            }
        }
    }
}

/// Optimize the bracket to a certain precision
pub trait BracketOptimizer {
    /// State of the optimizer, passed to `inspect_callback` at each evaluation
    type InternalState;

    /// Bracket the root of `f` between `upper` and `lower` up to `precision`
    ///
    /// Allows for inspecting the intermediate steps that bracketing algorithm takes via the
    /// `inspect_callback`.
    fn optimize<E>(
        &self,
        lower: Point,
        upper: Point,
        f: impl Fn(f64) -> Result<f64, E>,
        precision: Precision,
        inspect_callback: Option<&mut dyn FnMut(Self::InternalState)>,
    ) -> Result<BracketResult, BracketError<E>>;
}

/// Combination of `x` and `f(x)`
#[derive(Debug, Clone, Copy)]
pub struct Point {
    #[expect(missing_docs)]
    pub x: f64,
    #[expect(missing_docs)]
    pub f: f64,
}

/// Use inverse quadratic interpolation to obtain the next value at which to evaluate the function
pub struct InverseQuadratic {}

fn evaluate_bisect(x1: Point, x2: Point) -> f64 {
    0.5 * (x1.x + x2.x)
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
    assert!(upper >= 0.);
    assert!(lower >= 0.);

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

/// Intermediate state of [InverseQuadratic]
pub struct InverseQuadraticState {
    /// Current upper limit of the bracket
    pub upper: Point,
    /// Currennt lower limit of the bracket
    pub lower: Point,
    /// Last evaluation that is not `upper` or `lower`
    ///
    /// This is only set after the first evalution
    pub previous: Option<Point>,
    /// The next point that will be evaluated
    pub next_eval: f64,
}

impl BracketOptimizer for InverseQuadratic {
    type InternalState = InverseQuadraticState;

    fn optimize<E>(
        &self,
        mut lower: Point,
        mut upper: Point,
        f: impl Fn(f64) -> Result<f64, E>,
        precision: Precision,
        mut f_callback: Option<&mut dyn FnMut(Self::InternalState)>,
    ) -> Result<BracketResult, BracketError<E>> {
        assert!(lower.x.is_finite());
        assert!(lower.f.is_finite());
        assert!(upper.x.is_finite());
        assert!(upper.f.is_finite());

        if lower.f.signum() == upper.f.signum() {
            panic!("Upper and lower values in bracket have the same sign????");
        }

        let orig_lower = lower;
        let orig_upper = upper;
        let mut evals = 0;
        let mut previous: Option<Point> = None;
        let requested_ulp_precision = precision.as_ulp();

        loop {
            let x = match previous {
                None => evaluate_secant(lower, upper, 0.),
                Some(p) => {
                    // If the value of the determinant got worse, we might be too far away from the
                    // root, and neither inverse quadratic or secant will be accurate. Switch to
                    // bisection to prevent some pathalogical cases from converging very slowly
                    if (p.x > upper.x && p.f.abs() < upper.f.abs())
                        || (p.x < lower.x && p.f.abs() < lower.f.abs())
                    {
                        evaluate_bisect(lower, upper)
                    } else {
                        let x = evaluate_inverse_quadratic(lower, upper, p, 0.);

                        // In certain cases, the inverse quadratic fit may lead to estimated zeros
                        // outside the bracket. Secant method does not do this.
                        if x <= lower.x || x >= upper.x {
                            evaluate_secant(lower, upper, 0.)
                        } else {
                            x
                        }
                    }
                }
            };

            // Stopgap to prevent large determinants from causing problems
            let x = if !x.is_finite() {
                evaluate_bisect(lower, upper)
            } else {
                x
            };

            let x = match ensure_maximal_bracket(upper.x, lower.x, x, requested_ulp_precision) {
                ControlFlow::Break(x) => {
                    if lower.f.abs() >= orig_lower.f.abs() && upper.f.abs() >= orig_upper.f.abs() {
                        return Err(BracketError::Singularity);
                    }

                    return Ok(BracketResult {
                        lower,
                        upper,
                        root: x,
                        evals,
                    });
                }
                ControlFlow::Continue(x) => x,
            };

            if let Some(ref mut cb) = f_callback {
                cb(InverseQuadraticState {
                    lower,
                    upper,
                    previous,
                    next_eval: x,
                });
            }

            let next = Point { x, f: f(x)? };

            assert!(next.x.is_finite());
            assert!(next.f.is_finite());

            evals += 1;

            if upper.f.signum() == next.f.signum() {
                previous = Some(upper);
                upper = next;
            } else {
                previous = Some(lower);
                lower = next;
            }
        }
    }
}

/// Errors that may occur during bracketing
#[derive(Debug)]
pub enum BracketError<E> {
    /// Error occured in the function evaluation
    Eval(E),
    /// The bracket is around a singularity, not a root
    Singularity,
}

impl<E> From<E> for BracketError<E> {
    fn from(value: E) -> Self {
        Self::Eval(value)
    }
}

/// Extension trait to find sign swaps used for bracketing
///
/// This is intended to be implemented for iterators of numeric types
pub trait FilterSignSwap {
    /// The two sides of the bracket around the sign swap
    type Out;

    /// Filter for pairs of points that are adjacent and have different signs
    fn filter_sign_swap(self) -> impl Iterator<Item = Self::Out>;
}

impl<T: Iterator<Item = Point>> FilterSignSwap for T {
    type Out = (Point, Point);

    fn filter_sign_swap(self) -> impl Iterator<Item = (Point, Point)> {
        self.tuple_windows()
            .map(|(pair1, pair2)| {
                if pair1.f.signum() != pair2.f.signum() {
                    Some((pair1, pair2))
                } else {
                    None
                }
            })
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use itertools::Itertools;

    use crate::{
        bracket::Precision,
        dynamic_interface::{DifferenceSchemes, ErasedSolver},
        model::{
            interpolate::LinearInterpolator,
            polytrope::{IndexSegments, construct_polytrope},
        },
    };

    #[test]
    fn test_bracketing_convergence() {
        let poly = construct_polytrope(IndexSegments::central(3.), 5. / 3., 0.01);

        let determinant = ErasedSolver::new(
            &LinearInterpolator::new(&poly),
            1,
            0,
            DifferenceSchemes::Colloc4,
            &poly
                .segments
                .iter()
                .map(|x| x.dimensionless.r_coord.as_ref())
                .collect_vec(),
        );

        let points = rev_linspace(0.001, 1., 1000).chain(linspace(1., 500., 1000));

        assert_eq!(
            determinant
                .scan_and_optimize(
                    points,
                    Precision::ULP(const { NonZeroU64::new(1).unwrap() }),
                )
                .into_iter()
                .map(|res| res.evals)
                .collect_vec(),
            [
                7, 8, 9, 6, 7, 8, 6, 7, 7, 7, 9, 8, 6, 7, 10, 7, 7, 6, 6, 8, 10, 10, 10, 6, 8, 8,
                8, 13, 16, 20, 25, 31, 36, 40, 43, 43, 44, 42, 43, 43, 42, 44, 43, 43, 42, 44, 44,
                44, 43, 43, 43, 43, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 41, 44, 44, 44,
                44, 43, 44, 44, 42, 44, 44, 43, 44, 44, 44, 44, 43, 44, 43, 44, 44, 44, 44, 44, 44,
                44, 44, 43, 44, 44, 45, 44, 44, 47, 48, 47, 44, 13, 8, 5, 10, 12, 6, 7, 5, 6, 7, 6,
                6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 8, 9, 7, 6, 6, 5, 6, 6, 6, 6, 6, 6, 7, 6, 6,
                7, 5, 6, 6, 7, 7, 6, 6, 6, 6, 5, 6, 7, 7, 6, 6, 7, 8, 7, 7, 6, 7, 6, 7, 6, 6, 7, 7,
                8, 9, 6, 9, 7, 6, 10, 6, 8, 8, 8, 9, 9, 8, 6, 7, 6, 5, 9, 8, 7, 9, 9, 9, 9, 7, 6,
                5, 7, 9, 8, 6, 10, 8, 7, 7, 7, 6, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 8, 6, 5, 7, 7, 7,
                7, 7, 8, 7, 8, 6, 7, 7, 7, 7, 7, 6, 7, 7, 7, 9, 6, 7, 7, 8, 8, 8, 6, 6, 8, 7, 7, 7,
                7, 7, 8, 6, 7, 7, 7, 7, 8, 9, 6, 8, 7, 7, 7, 9, 7, 7, 8, 7, 7, 7, 7, 8, 7, 6, 8, 6,
                6, 6, 9, 7, 7, 9, 7, 7, 8, 7, 8, 8, 8, 7, 7, 7, 7, 7, 8, 7, 6, 5, 8, 6, 6, 7, 7, 7,
                8, 7, 8, 8, 9, 7, 9, 8, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 7, 6, 6, 7, 7,
                7, 7, 8, 6, 8, 8, 9, 11, 17, 20, 23, 24, 25, 29, 35, 39, 42, 44, 44, 43, 43, 40,
                44, 43, 43, 44, 44, 43, 42, 44, 44, 42, 38, 44, 43, 42, 44, 43, 42, 44, 43, 43, 44,
                43, 42, 44, 43, 40, 44, 43, 44, 44, 42, 44, 43, 44, 44, 43, 44, 43, 44, 43, 42, 44,
                42, 44, 43, 44, 43, 44, 43, 40, 44, 42, 43, 41, 43, 42, 44, 42, 43, 41, 43, 39, 43,
                44, 43, 44, 43, 44, 43, 44, 41, 43, 44, 44, 44, 42, 44, 44, 43, 44, 42, 44, 44, 43
            ]
        );
    }

    fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
    }

    fn rev_linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        linspace(1. / lower, 1. / upper, n).map(|x| 1. / x)
    }
}
