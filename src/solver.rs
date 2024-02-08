use std::cmp::min;

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

const fn to_band_coord(ku: usize, kl: usize, i: usize, j: usize) -> usize {
    let bandwith = 2 * kl + ku + 1;

    bandwith * j + kl + ku + i - j
}

pub const fn calc_n_bands<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    2 * calc_kl::<N, N_INNER, N_OUTER>() + calc_ku::<N, N_INNER, N_OUTER>() + 1
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

pub fn decompose_system_matrix<
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
) -> Result<DecomposedSystemMatrix, ()>
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
    let mut band_storage = vec![0.0; calc_n_bands::<N, N_INNER, N_OUTER>() * alen];

    let (storage, _): (&mut [[f64; calc_n_bands::<N, N_INNER, N_OUTER>()]], _) =
        band_storage.as_chunks_mut();

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
            band_storage.as_mut(),
            calc_n_bands::<N, N_INNER, N_OUTER>() as i32,
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
