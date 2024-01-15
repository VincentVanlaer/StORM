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

pub(crate) fn bracket_search<
    const N: usize,
    const N_INNER: usize,
    const N_OUTER: usize,
    const ORDER: usize,
    G: ?Sized,
    I: System<f64, G, N, N_INNER, N_OUTER, ORDER>,
    S: Stepper<f64, N, ORDER>,
    Searcher: BracketSearcher,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    lower: f64,
    upper: f64,
    searcher: &Searcher,
) -> Result<f64, ()>
where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
    [(); calc_n_bands::<N, N_INNER, N_OUTER>()]: Sized,
{
    let lower_value = determinant(system, stepper, grid, lower);
    let upper_value = determinant(system, stepper, grid, upper);

    let mut searcher = searcher.init(lower, lower_value, upper, upper_value)?;

    loop {
        let value = determinant(system, stepper, grid, searcher.point());

        searcher = match searcher.next(value) {
            BracketResult::Completed(lower, upper) => return Ok(0.5 * (lower + upper)),
            BracketResult::NextPoint(s) => s,
        };
    }
}

pub(crate) trait BracketSearcher {
    type Iterator: BracketSearchIterator;

    fn init(
        &self,
        lower: f64,
        lower_value: f64,
        upper: f64,
        upper_value: f64,
    ) -> Result<Self::Iterator, ()>;
}

pub(crate) enum BracketResult<I: BracketSearchIterator> {
    NextPoint(I),
    Completed(f64, f64),
}

pub(crate) trait BracketSearchIterator: Sized {
    fn point(&self) -> f64;
    fn next(self, value: f64) -> BracketResult<Self>;
}

pub(crate) struct BisectionIterator {
    rel_epsilon: f64,
    upper: f64,
    upper_value: f64,
    lower: f64,
    lower_value: f64,
}

impl BracketSearchIterator for BisectionIterator {
    fn point(&self) -> f64 {
        self.lower + 0.5 * (self.upper - self.lower)
    }

    fn next(mut self, value: f64) -> BracketResult<Self> {
        if self.upper_value.signum() == value.signum() {
            self.upper = self.point();
            self.upper_value = value;
        } else {
            self.lower = self.point();
            self.lower_value = value;
        }

        if (self.upper - self.lower) <= (self.upper.abs().max(self.lower.abs())) * self.rel_epsilon
        {
            BracketResult::Completed(self.lower, self.upper)
        } else {
            BracketResult::NextPoint(self)
        }
    }
}

pub(crate) struct Bisection {
    pub rel_epsilon: f64,
}

impl BracketSearcher for Bisection {
    type Iterator = BisectionIterator;

    fn init(
        &self,
        lower: f64,
        lower_value: f64,
        upper: f64,
        upper_value: f64,
    ) -> Result<Self::Iterator, ()> {
        if lower_value.signum() == upper_value.signum() {
            Err(())
        } else {
            Ok(BisectionIterator {
                rel_epsilon: self.rel_epsilon,
                lower,
                lower_value,
                upper,
                upper_value,
            })
        }
    }
}

pub(crate) struct BrentIterator {
    rel_epsilon: f64,
    upper: f64,
    upper_value: f64,
    lower: f64,
    lower_value: f64,
    mid: f64,
    mid_value: f64,
    mflag: Option<f64>,
    s: f64,
}

impl BrentIterator {
    fn compute_point(&mut self) {
        let mut s = if self.lower_value != self.mid_value && self.upper_value == self.mid_value {
            let s_lower = self.lower * self.upper_value * self.mid_value
                / ((self.lower_value - self.upper_value) * (self.lower_value - self.mid_value));
            let s_upper = self.upper * self.lower_value * self.mid_value
                / ((self.upper_value - self.lower_value) * (self.upper_value - self.mid_value));
            let s_mid = self.mid * self.lower_value * self.upper_value
                / ((self.mid_value - self.lower_value) * (self.mid_value - self.upper_value));

            s_lower + s_upper + s_mid
        } else {
            self.upper
                - self.upper_value * (self.upper - self.lower)
                    / (self.upper_value - self.lower_value)
        };

        let m = match self.mflag {
            Some(d) => d,
            None => self.upper,
        };

        if ((s < (3. * self.lower + self.upper) / 4.) && (s < self.upper))
            || ((s > (3. * self.lower + self.upper) / 4.) && (s > self.upper))
            || (2. * (s - self.upper).abs() >= (m - self.mid).abs())
            || ((m - self.mid).abs() <= self.rel_epsilon * f64::max(m.abs(), self.mid.abs()))
        {
            s = self.lower + 0.5 * (self.upper - self.lower);
            self.mflag = Some(self.mid);
        } else {
            self.mflag = None;
        }

        self.s = s;
    }
}

impl BracketSearchIterator for BrentIterator {
    fn point(&self) -> f64 {
        self.s
    }

    fn next(mut self, value: f64) -> BracketResult<Self> {
        self.mid = self.upper;
        self.mid_value = self.upper_value;

        if self.lower_value.signum() == value.signum() {
            self.upper = self.s;
            self.upper_value = value;
        } else {
            self.lower = self.s;
            self.lower_value = value;
        }

        if self.upper_value.abs() < self.lower_value.abs() {
            (self.lower, self.upper, self.upper_value, self.lower_value) =
                (self.upper, self.lower, self.lower_value, self.upper_value);
        }

        if (self.upper - self.lower) <= (self.upper.abs().max(self.lower.abs())) * self.rel_epsilon
        {
            BracketResult::Completed(self.lower, self.upper)
        } else {
            self.compute_point();
            BracketResult::NextPoint(self)
        }
    }
}

pub(crate) struct Brent {
    pub rel_epsilon: f64,
}

impl BracketSearcher for Brent {
    type Iterator = BrentIterator;

    fn init(
        &self,
        mut lower: f64,
        mut lower_value: f64,
        mut upper: f64,
        mut upper_value: f64,
    ) -> Result<Self::Iterator, ()> {
        if lower_value.signum() == upper_value.signum() {
            Err(())
        } else {
            if upper_value.abs() < lower_value.abs() {
                (lower, upper, upper_value, lower_value) = (upper, lower, lower_value, upper_value);
            }
            let mut iterator = BrentIterator {
                rel_epsilon: self.rel_epsilon,
                lower,
                lower_value,
                upper,
                upper_value,
                mflag: None,
                mid: lower,
                mid_value: lower_value,
                s: f64::NAN,
            };

            iterator.compute_point();

            Ok(iterator)
        }
    }
}
