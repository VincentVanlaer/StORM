#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(min_specialization)]
#![feature(portable_simd)]
#![allow(incomplete_features)]

use color_eyre::Result;
use hdf5::File as hdf5File;
use linalg::Matrix;
use model::StellarModel;
use num::Float;
use std::fmt::Display;

use crate::{jacobian::NonRotating1D, solver::determinant, stepper::MagnusGL6};

mod jacobian;
mod linalg;
mod model;
mod solver;
mod stepper;

extern crate blas_src;
extern crate lapack_src;

fn print_matrix<T: Float + Display, const ROWS: usize, const COLUMNS: usize>(
    matrix: &Matrix<T, ROWS, COLUMNS>,
) where
    [(); ROWS * COLUMNS]: Sized,
{
    for i in 0..COLUMNS {
        for j in 0..ROWS {
            print!("{:>20.10}", matrix[j][i]);
        }
        println!();
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let f = hdf5File::open("Z0.016_M11.50_logD3.00_aov0.000_fov0.020_mn2186-MS.GSM")?;

    let model = StellarModel::from_gsm(&f)?;
    let interpolator = NonRotating1D::from_model(&model, 1)?;

    println!("{}", model.r_coord.len());

    const STEPS: usize = 6000;
    const FREQS: usize = 1000;

    let grid = (0..STEPS + 1)
        .map(|n| 1. / STEPS as f64 * n as f64)
        .collect();
    let mut dets = [0.0; FREQS];

    for l in 0..FREQS {
        let freq = 1.62 + (1. / FREQS as f64) * l as f64;
        let det = determinant(&interpolator, &MagnusGL6 {}, &grid, freq);
        dets[l] = det;
    }

    for (i, det) in dets.iter().enumerate() {
        let freq = 1.62 + (1. / FREQS as f64) * i as f64;
        println!("{freq:.3}: {det}");
    }

    Ok(())
}
