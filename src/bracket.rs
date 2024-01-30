pub(crate) struct BracketResult {
    pub freq: f64,
    pub evals: u64,
}

pub(crate) trait BracketSearcher<E> {
    fn search<F: Fn(f64) -> Result<f64, E>>(
        &self,
        lower: Point,
        upper: Point,
        f: F,
    ) -> Result<BracketResult, E>;
}

pub(crate) struct Bisection {
    pub rel_epsilon: f64,
}

impl<E> BracketSearcher<E> for Bisection {
    fn search<F: Fn(f64) -> Result<f64, E>>(
        &self,
        mut lower: Point,
        mut upper: Point,
        f: F,
    ) -> Result<BracketResult, E> {
        if lower.f.signum() == upper.f.signum() {
            panic!("Upper and lower values in bracket have the same sign????");
        }

        let mut evals = 0;

        loop {
            let delta = (upper.x - lower.x).abs();
            let x = lower.x + 0.5 * (upper.x - lower.x);

            if delta <= f64::max(upper.x.abs(), lower.x.abs()) * (self.rel_epsilon + f64::EPSILON) {
                return Ok(BracketResult { freq: x, evals });
            }

            let next = Point { x, f: f(x)? };

            evals += 1;

            if upper.f.signum() == next.f.signum() {
                upper = next;
            } else {
                lower = next;
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Point {
    pub x: f64,
    pub f: f64,
}

pub(crate) struct Brent {
    pub rel_epsilon: f64,
}

impl<E> BracketSearcher<E> for Brent {
    fn search<F: Fn(f64) -> Result<f64, E>>(
        &self,
        lower: Point,
        upper: Point,
        f: F,
    ) -> Result<BracketResult, E> {
        if lower.f.signum() == upper.f.signum() {
            panic!("Upper and lower values in bracket have the same sign????");
        }

        let mut previous = lower;
        let mut current = upper;
        let mut counterpoint = current;
        let mut d = 0.0;
        let mut e = 0.0;
        let mut evals = 0;
        let mut method: &str = "Initial";

        loop {
            if counterpoint.f.signum() == current.f.signum() {
                counterpoint = previous;
                d = current.x - previous.x;
                e = d;
            }

            if counterpoint.f.abs() < current.f.abs() {
                previous = current;
                (counterpoint, current) = (current, counterpoint);
            }

            // dbg!((method, &current, &previous, &counterpoint, d, e));

            let accuracy = 0.5 * (counterpoint.x - current.x);
            let tolerance =
                (f64::EPSILON + self.rel_epsilon) * f64::max(current.x.abs(), counterpoint.x.abs());

            if accuracy.abs() <= tolerance || current.f == 0.0 {
                return Ok(BracketResult {
                    freq: current.x + accuracy,
                    evals,
                });
            }

            if e.abs() >= tolerance && previous.f.abs() >= current.f.abs() {
                let slope = current.f / previous.f;
                let mut p;
                let mut q;

                if previous.f == counterpoint.f {
                    p = 2.0 * accuracy * slope;
                    q = 1.0 - slope;
                    method = "Secant";
                } else {
                    let slope_ac = previous.f / counterpoint.f;
                    let slope_bc = current.f / counterpoint.f;

                    p = slope
                        * (2.0 * accuracy * slope_ac * (slope_ac - slope_bc)
                            - (current.x - previous.x) * (slope_bc - 1.0));
                    q = (slope_ac - 1.0) * (slope_bc - 1.0) * (slope - 1.0);
                    method = "QIP";
                }

                if p > 0.0 {
                    q = -q;
                } else {
                    p = -p;
                }

                let min1 = 3.0 * accuracy * q - (tolerance * q).abs();
                let min2 = (e * q).abs();

                if 2.0 * p < f64::min(min1, min2) {
                    e = d;
                    d = p / q;
                } else {
                    d = accuracy;
                    e = d;
                    method = "Bisect (rejected)";
                }
            } else {
                d = accuracy;
                e = d;
                method = "Bisect (wrong direction or tolerance)";
            }

            previous = current;

            current.x += d;
            current.f = f(current.x)?;
            evals += 1;
        }
    }
}
