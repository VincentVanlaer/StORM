use core::panic;
use std::mem::transmute;

use lapack::dgbtrf;

use crate::{stepper::Stepper, system::System};

const fn calc_ku<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    N - 1 + (N - N_INNER)
}

const fn calc_kl<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    if N >= N_OUTER {
        N_INNER - 1 + N
    } else {
        N_INNER - 1 + N_OUTER
    }
}

pub(crate) const fn calc_n_bands<const N: usize, const N_INNER: usize, const N_OUTER: usize>(
) -> usize {
    2 * calc_kl::<N, N_INNER, N_OUTER>() + calc_ku::<N, N_INNER, N_OUTER>() + 1
}

pub(crate) fn determinant<
    const N: usize,
    const N_INNER: usize,
    const N_OUTER: usize,
    const ORDER: usize,
    G: ?Sized,
    I: System<f64, G, N, N_INNER, N_OUTER, ORDER>,
    S: Stepper<f64, N, ORDER>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: f64,
) -> f64
where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
    [(); calc_n_bands::<N, N_INNER, N_OUTER>()]: Sized,
{
    let iterator = system.evaluate_moments(grid, frequency);
    let alen: usize = (iterator.len() + 1) * N;
    let ku: usize = calc_ku::<N, N_INNER, N_OUTER>();
    let kl: usize = calc_kl::<N, N_INNER, N_OUTER>();
    let mut storage = vec![[0.0; calc_n_bands::<N, N_INNER, N_OUTER>()]; alen];

    let mut ipiv = vec![0; alen];
    let mut info: i32 = 0;
    let outer_boundary = system.outer_boundary(frequency);
    let inner_boundary = system.inner_boundary(frequency);

    for j in 0..N {
        for k in 0..N_INNER {
            storage[j][kl + ku + k - j] = inner_boundary[j][k];
        }
        for k in 0..N_OUTER {
            storage[alen - N + j][kl + ku + N - j - N_OUTER + k] = outer_boundary[j][k];
        }
    }

    for (i, step) in iterator.map(|x| stepper.step(x)).enumerate() {
        for j in 0..N {
            for k in 0..N {
                storage[i * N + j][kl + ku + k - j + N_INNER] = step.left[j][k];
                storage[(i + 1) * N + j][kl + ku + k - j - N + N_INNER] = step.right[j][k];
            }
        }
    }

    unsafe {
        dgbtrf(
            alen as i32,
            alen as i32,
            kl as i32,
            ku as i32,
            transmute(storage.as_mut_slice()),
            calc_n_bands::<N, N_INNER, N_OUTER>() as i32,
            ipiv.as_mut_slice(),
            &mut info,
        )
    };

    if info < 0 {
        panic!("dgbtrf failed")
    }

    let mut det = 1.0;

    for i in 0..alen {
        det *= storage[i][kl + ku];
    }

    let mut sgn = 1;

    for i in 0..alen {
        if ipiv[i] != (i + 1) as i32 {
            sgn *= -1;
        }
    }

    sgn as f64 * det
}

pub(crate) struct BracketResult {
    pub freq: f64,
    pub evals: u64,
}

pub(crate) trait BracketSearcher {
    fn search<F: Fn(f64) -> f64>(
        &self,
        lower: Point,
        upper: Point,
        f: F,
    ) -> Result<BracketResult, ()>;
}

pub(crate) struct Bisection {
    pub rel_epsilon: f64,
}

impl BracketSearcher for Bisection {
    fn search<F: Fn(f64) -> f64>(
        &self,
        mut lower: Point,
        mut upper: Point,
        f: F,
    ) -> Result<BracketResult, ()> {
        if lower.f.signum() == upper.f.signum() {
            return Err(());
        }

        let mut evals = 0;

        loop {
            let delta = (upper.x - lower.x).abs();
            let x = lower.x + 0.5 * (upper.x - lower.x);

            if delta <= f64::max(upper.x.abs(), lower.x.abs()) * (self.rel_epsilon + f64::EPSILON) {
                return Ok(BracketResult { freq: x, evals });
            }

            let next = Point { x, f: f(x) };

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

impl BracketSearcher for Brent {
    fn search<F: Fn(f64) -> f64>(
        &self,
        lower: Point,
        upper: Point,
        f: F,
    ) -> Result<BracketResult, ()> {
        if lower.f.signum() == upper.f.signum() {
            return Err(());
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
            current.f = f(current.x);
            evals += 1;
        }
    }
}
