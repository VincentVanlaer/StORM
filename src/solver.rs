use std::mem::transmute;

use lapack::dgbtrf;

use crate::{jacobian::System, stepper::Step};

const fn calc_ku<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    N - 1
}

const fn calc_kl<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    if N >= N_OUTER {
        N_INNER - 1 + N
    } else {
        N_INNER - 1 + N_OUTER
    }
}

pub(crate) const fn calc_n_bands<const N: usize, const N_INNER: usize, const N_OUTER: usize>() -> usize {
    2 * calc_kl::<N, N_INNER, N_OUTER>() + calc_ku::<N, N_INNER, N_OUTER>() + 1
}

pub(crate) fn determinant<
    const N: usize,
    const N_INNER: usize,
    const N_OUTER: usize,
    I: System<f64, N, N_INNER, N_OUTER>,
    S: Step<f64, N, I>,
>(
    system: &I,
    stepper: &S,
    grid: &Vec<f64>,
    frequency: f64,
) -> f64
where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
    [(); calc_n_bands::<N, N_INNER, N_OUTER>()]: Sized,
{
    let ku: usize = calc_ku::<N, N_INNER, N_OUTER>();
    let kl: usize = calc_kl::<N, N_INNER, N_OUTER>();
    let alen: usize = grid.len() * N;
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
            storage[alen - 4 + j][kl + ku + k - j + 2] = outer_boundary[j][k];
        }
    }

    for (i, pos) in grid.windows(2).enumerate() {
        let matrix = stepper.step(system, pos[0], pos[1], frequency);

        for j in 0..4 {
            for k in 0..4 {
                storage[i * 4 + j][kl + ku + k - j + 2] = matrix[j][k];
            }
            storage[(i + 1) * 4 + j][kl + ku - 2] = -1.0;
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
