use core::panic;
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

pub(crate) const fn calc_n_bands<const N: usize, const N_INNER: usize, const N_OUTER: usize>(
) -> usize {
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
            storage[alen - N + j][kl + ku + N - j - N_OUTER + k] = outer_boundary[j][k];
        }
    }

    for (i, pos) in grid.windows(2).enumerate() {
        let matrix = stepper.step(system, pos[0], pos[1], frequency);

        for j in 0..N {
            for k in 0..N {
                storage[i * N + j][kl + ku + k - j + N_INNER] = matrix[j][k];
            }
            storage[(i + 1) * N + j][kl + ku - N + N_INNER] = -1.0;
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


const DEBUG_BRACKETS: bool = true;

pub(crate) fn bracket_search<
    const N: usize,
    const N_INNER: usize,
    const N_OUTER: usize,
    I: System<f64, N, N_INNER, N_OUTER>,
    S: Step<f64, N, I>,
>(
    system: &I,
    stepper: &S,
    grid: &Vec<f64>,
    mut lower: f64,
    mut upper: f64,
) -> f64
where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
    [(); calc_n_bands::<N, N_INNER, N_OUTER>()]: Sized,
{
    const KAPPA1: f64 = 0.01;
    const KAPPA2: f64 = 2.618;
    const N0: i32 = 2;

    let epsilon = f64::EPSILON * upper.abs();
    let n_bisect = (((upper - lower) / (2.0 * epsilon)).log2() as i32) + N0;

    let mut j = 0;

    let mut f_lower = determinant(system, stepper, grid, lower);
    let mut f_upper = determinant(system, stepper, grid, upper);

    if DEBUG_BRACKETS {
        println!("Max number of iterations: {n_bisect}, epsilon = {epsilon}");
        println!(
            "Initial bracket [{}, {}], delta = {} with values {} and {}",
            lower,
            upper,
            upper - lower,
            f_lower,
            f_upper
        );
    }

    if f_lower.signum() == f_upper.signum() {
        panic!("Lower and upper values in bracket search have same values");
    }

    loop {
        let x12 = 0.5 * (lower + upper);
        let xf = (upper * f_lower - lower * f_upper) / (f_lower - f_upper);

        let sigma = (x12 - xf).signum();
        let delta = KAPPA1 * (upper - lower).powf(KAPPA2);
        let r = epsilon * 2.0_f64.powi(n_bisect - j) - (upper - lower) / 2.0;

        let xt = if delta < (xf - x12).abs() {
            xf + sigma * delta
        } else {
            x12
        };

        let mut x_itp = xt.min(x12 + r).max(x12 - r);

        if x_itp == lower || x_itp == upper {
            // Floating point shenanigans
            x_itp = x12;
        }

        let f_itp = determinant(system, stepper, grid, x_itp);

        j += 1;

        if f_itp == 0.0 {
            lower = x_itp;
            upper = x_itp;
            break;
        }

        if f_itp.signum() == f_upper.signum() {
            f_upper = f_itp;
            upper = x_itp;
        } else {
            f_lower = f_itp;
            lower = x_itp;
        }

        if (upper - lower).abs() < 2.0 * epsilon {
            break;
        }

        if DEBUG_BRACKETS {
            println!(
                "Iteration done, bracket refined to [{}, {}], delta = {} with values {} and {}",
                lower,
                upper,
                upper - lower,
                f_lower,
                f_upper
            );
        }
    }

    if DEBUG_BRACKETS {
        println!("Total number of iterations executed: {j}");
    }

    0.5 * (upper + lower)
}
