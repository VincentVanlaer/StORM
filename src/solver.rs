use crate::{stepper::Stepper, system::System};

pub(crate) struct UpperResult {
    data: Box<[f64]>,
    n: usize,
    n_systems: usize,
}

pub(crate) fn determinant<
    const N: usize,
    const N_INNER: usize,
    const ORDER: usize,
    G: ?Sized,
    I: System<f64, G, N, N_INNER, ORDER>,
    S: Stepper<f64, N, ORDER>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: f64,
) -> f64
where
    [(); { N - N_INNER } * N]: Sized,
    [(); 2 * N]: Sized,
    [(); N + N_INNER]: Sized,
{
    determinant_inner(system, stepper, grid, frequency, &mut ())
}

pub(crate) fn determinant_with_upper<
    const N: usize,
    const N_INNER: usize,
    const ORDER: usize,
    G: ?Sized,
    I: System<f64, G, N, N_INNER, ORDER>,
    S: Stepper<f64, N, ORDER>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: f64,
    upper: &mut UpperResult,
) -> f64
where
    [(); { N - N_INNER } * N]: Sized,
    [(); 2 * N]: Sized,
    [(); N + N_INNER]: Sized,
{
    assert_eq!(upper.n, N);
    assert_eq!(upper.n_systems, system.len(grid));

    determinant_inner(system, stepper, grid, frequency, upper)
}

fn gauss(lower: usize, upper: usize) -> usize {
    (upper - lower + 1) * (upper + lower) / 2
}

// Data storage of UpperResult and backwards substitution example. The different indices i, j, and
// k refer to the variables in UpperResult::eigenvectors. The goal is to determine the indices in
// the top row, as those give which element of the eigenvector needs to be multiplied with the
// upper block triangular matrix. The data is indexed row by row, starting from the top left. Note
// that this is *back* substitution so the direction of this iterator is reversed.
//
// 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ i   k                j
// 1 * * * * * * *                 ┃ 3 ┓    * 6 5 4 3 2 1 0
// 0 1 * * * * * *                 ┃ 2 ┣ 2    * 5 4 3 2 1 0
// 0 0 1 * * * * *                 ┃ 1 ┃        * 4 3 2 1 0
// 0 0 0 1 * * * *                 ┃ 0 ┛          * 3 2 1 0
// 0 0 0 0 1 * * * * * * *         ┃ 3 ┓            * 6 5 4 3 2 1 0
// 0 0 0 0 0 1 * * * * * *         ┃ 2 ┣ 1            * 5 4 3 2 1 0
//         0 0 1 * * * * *         ┃ 1 ┃                * 4 3 2 1 0
//         0 0 0 1 * * * *         ┃ 0 ┛                  * 3 2 1 0
//         0 0 0 0 1 * * * * * * * ┃ 3 ┓                    * 6 5 4 3 2 1 0
//         0 0 0 0 0 1 * * * * * * ┃ 2 ┣ 0                    * 5 4 3 2 1 0
//                 0 0 1 * * * * * ┃ 1 ┃                        * 4 3 2 1 0
//                 0 0 0 1 * * * * ┃ 0 ┛                          * 3 2 1 0
//                 0 0 0 0 1 * * * ┃ 3 ┓                            * 2 1 0
//                 0 0 0 0 0 1 * * ┃ 2 ┣ special case                 * 1 0
//                         0 0 1 * ┃ 1 ┃ has less data to the           * 0
//                         0 0 0 ^ ┃ 0 ┛ right due to edge of matrix      *

impl UpperResult {
    pub(crate) fn new<
        const N: usize,
        const N_INNER: usize,
        const ORDER: usize,
        G: ?Sized,
        I: System<f64, G, N, N_INNER, ORDER>,
    >(
        system: &I,
        grid: &G,
    ) -> UpperResult {
        let n_systems = system.len(grid);
        UpperResult {
            data: vec![f64::NAN; n_systems * gauss(N, 2 * N - 1) + gauss(1, N - 1)].into_boxed_slice(),
            n: N,
            n_systems,
        }
    }

    pub(crate) fn eigenvectors(&self) -> Vec<f64> {
        let mut eigenvectors = vec![0.0; self.n_systems * self.n];
        let len = eigenvectors.len();

        eigenvectors[len - 1] = 1.;

        let mut data_iter = self.data.iter().rev();

        for i in 1..self.n {
            let mut next_val = 0.;
            for j in 0..i {
                next_val -= data_iter.next().unwrap() * eigenvectors[eigenvectors.len() - j - 1]
            }
            eigenvectors[len - i - 1] = next_val;
        }

        for k in 0..(self.n_systems - 1) {
            for i in 0..self.n {
                let mut next_val = 0.;
                for j in 0..(i + self.n) {
                    next_val -= data_iter.next().unwrap()
                        * eigenvectors[eigenvectors.len() - j - k * self.n - 1]
                }
                eigenvectors[len - i - (k + 1) * self.n - 1] = next_val;
            }
        }

        eigenvectors
    }
}

trait SetUpperResult {
    fn set(&mut self, point: usize, k: usize, i: usize, val: f64);
}

impl SetUpperResult for UpperResult {
    fn set(&mut self, point: usize, k: usize, i: usize, val: f64) {
        if point != self.n_systems {
            self.data[point * gauss(self.n, 2 * self.n - 1)
                + gauss(2 * self.n - 1 - k, 2 * self.n - 1)
                - (2 * self.n - 1 - k)
                + (i - 1)] = val;
        } else {
            self.data[point * gauss(self.n, 2 * self.n - 1) + gauss(self.n - 1 - k, self.n - 1)
                - (self.n - 1 - k)
                + (i - 1)] = val;
        }
    }
}

impl SetUpperResult for () {
    fn set(&mut self, _point: usize, _k: usize, _i: usize, _val: f64) {}
}

fn determinant_inner<
    const N: usize,
    const N_INNER: usize,
    const ORDER: usize,
    G: ?Sized,
    I: System<f64, G, N, N_INNER, ORDER>,
    S: Stepper<f64, N, ORDER>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: f64,
    upper: &mut impl SetUpperResult,
) -> f64
where
    [(); { N - N_INNER } * N]: Sized,
    [(); 2 * N]: Sized,
    [(); N + N_INNER]: Sized,
{
    let iterator = system.evaluate_moments(grid, frequency);
    let total_steps = iterator.len();
    assert_eq!(total_steps, system.len(grid));
    let outer_boundary = system.outer_boundary(frequency);
    let inner_boundary = system.inner_boundary(frequency);

    let mut bands = [[0.0; 2 * N]; N + N_INNER];

    for i in 0..N_INNER {
        for j in 0..N {
            bands[i][j] = inner_boundary[i][j];
        }
    }

    let mut det = 1.0;
    for (n_step, step) in iterator.map(|x| stepper.step(x)).enumerate() {
        assert!(n_step < total_steps);
        for i in 0..N {
            for j in 0..N {
                bands[i + 2][j] = step.left[i][j];
                bands[i + 2][j + N] = step.right[i][j];
            }
        }

        for k in 0..N {
            let mut max_idx = 0;
            let mut max_val: f64 = 0.;

            for i in k..(N + N_INNER) {
                if bands[i][k].abs() > max_val.abs() {
                    max_idx = i;
                    max_val = bands[i][k];
                }
            }

            let pivot;

            if max_idx != k {
                pivot = bands[max_idx];
                bands[max_idx] = bands[k];
                det *= -1.;
            } else {
                pivot = bands[k];
            }

            det *= pivot[k];

            for i in (k + 1)..(2 * N) {
                upper.set(n_step, k, i - k, pivot[i] / pivot[k]);
            }

            for i in (k + 1)..(N + N_INNER) {
                let m = bands[i][k] / pivot[k];
                for j in 0..(2 * N) {
                    bands[i][j] -= pivot[j] * m;
                }
            }
        }

        for i in 0..N_INNER {
            *bands[i].split_array_mut::<N>().0 = *bands[i + N].rsplit_array_ref::<N>().1;
            *bands[i].rsplit_array_mut::<N>().1 = [0.0; _];
        }
    }

    // Outer boundary
    for i in 0..(N - N_INNER) {
        for j in 0..N {
            bands[N_INNER + i][j] = outer_boundary[i][j];
        }
    }

    for k in 0..(N - 1) {
        let mut max_idx = 0;
        let mut max_val: f64 = 0.;

        for i in k..N {
            if bands[i][k].abs() > max_val.abs() {
                max_idx = i;
                max_val = bands[i][k];
            }
        }

        if max_idx != k {
            (bands[max_idx], bands[k]) = (bands[k], bands[max_idx]);
            det *= -1.;
        }

        let pivot = bands[k][k];

        for j in 0..N {
            bands[k][j] /= pivot;
        }

        for i in (k + 1)..N {
            upper.set(total_steps, k, i - k, bands[k][i]);
        }

        det *= pivot;

        for i in (k + 1)..N {
            let m = bands[i][k];
            for j in 0..N {
                bands[i][j] -= bands[k][j] * m;
            }
        }
    }

    det * bands[N - 1][N - 1]
}
