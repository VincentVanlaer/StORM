use std::cmp::min;

use lapack::dgbtrf;

use crate::{stepper::Stepper, system::System};

const fn calc_ku<const N: usize, const N_INNER: usize>() -> usize {
    N - 1 + (N - N_INNER)
}

const fn calc_kl<const N: usize, const N_INNER: usize>() -> usize {
    N_INNER - 1 + N
}

const fn to_band_coord(ku: usize, kl: usize, i: usize, j: usize) -> usize {
    let bandwith = 2 * kl + ku + 1;

    bandwith * j + kl + ku + i - j
}

pub const fn calc_n_bands<const N: usize, const N_INNER: usize>() -> usize {
    2 * calc_kl::<N, N_INNER>() + calc_ku::<N, N_INNER>() + 1
}

pub struct DecomposedSystemMatrix {
    band_storage: Vec<f64>,
    ku: usize,
    kl: usize,
    ipiv: Vec<i32>,
}

impl DecomposedSystemMatrix {
    pub fn determinant(&self) -> f64 {
        let alen = self.ipiv.len();

        let mut det = 1.0;

        for i in 0..alen {
            det *= self.band_storage[to_band_coord(self.ku, self.kl, i, i)];
        }

        let mut sgn = 1;

        for i in 0..alen {
            if self.ipiv[i] != (i + 1) as i32 {
                sgn *= -1;
            }
        }

        sgn as f64 * det
    }

    fn data(&self, i: usize, j: usize) -> f64 {
        self.band_storage[to_band_coord(self.ku, self.kl, i, j)]
    }

    pub fn eigenvector(&self) -> Vec<f64> {
        let alen = self.ipiv.len();
        let u_diagonals = self.kl + self.ku + 1;

        let mut eigenvector = vec![0.0; alen];

        eigenvector[alen - 1] = 1.;

        for i in (0..=(alen - 2)).rev() {
            let mut sum = 0.;

            for j in (i + 1)..min(alen, i + u_diagonals) {
                sum += eigenvector[j] * self.data(i, j);
            }

            eigenvector[i] = -1. / self.data(i, i) * sum;
        }

        eigenvector
    }
}

pub fn direct_determinant<
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
    let iterator = system.evaluate_moments(grid, frequency);
    let outer_boundary = system.outer_boundary(frequency);
    let inner_boundary = system.inner_boundary(frequency);

    let mut bands = [[0.0; 2 * N]; N + N_INNER];

    for i in 0..N_INNER {
        for j in 0..N {
            bands[i][j] = inner_boundary[i][j];
        }
    }

    let mut det = 1.0;
    for step in iterator.map(|x| stepper.step(x)) {
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

pub fn decompose_system_matrix<
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
) -> Result<DecomposedSystemMatrix, ()>
where
    [(); { N - N_INNER } * N]: Sized,
    [(); calc_n_bands::<N, N_INNER>()]: Sized,
{
    let iterator = system.evaluate_moments(grid, frequency);
    let alen: usize = (iterator.len() + 1) * N;
    let ku: usize = calc_ku::<N, N_INNER>();
    let kl: usize = calc_kl::<N, N_INNER>();
    let mut band_storage = vec![0.0; calc_n_bands::<N, N_INNER>() * alen];

    let (storage, _): (&mut [[f64; calc_n_bands::<N, N_INNER>()]], _) =
        band_storage.as_chunks_mut();

    let mut ipiv = vec![0; alen];
    let mut info: i32 = 0;
    let outer_boundary = system.outer_boundary(frequency);
    let inner_boundary = system.inner_boundary(frequency);

    for j in 0..N {
        for k in 0..N_INNER {
            storage[j][kl + ku + k - j] = inner_boundary[k][j];
        }
        for k in 0..(N - N_INNER) {
            storage[alen - N + j][kl + ku + N - j - (N - N_INNER) + k] = outer_boundary[k][j];
        }
    }

    for (i, step) in iterator.map(|x| stepper.step(x)).enumerate() {
        for j in 0..N {
            for k in 0..N {
                storage[i * N + j][kl + ku + k - j + N_INNER] = step.left[k][j];
                storage[(i + 1) * N + j][kl + ku + k - j - N + N_INNER] = step.right[k][j];
            }
        }
    }

    unsafe {
        dgbtrf(
            alen as i32,
            alen as i32,
            kl as i32,
            ku as i32,
            band_storage.as_mut(),
            calc_n_bands::<N, N_INNER>() as i32,
            ipiv.as_mut_slice(),
            &mut info,
        )
    };

    if info < 0 {
        Err(())
    } else {
        Ok(DecomposedSystemMatrix {
            band_storage,
            ipiv,
            ku,
            kl,
        })
    }
}
