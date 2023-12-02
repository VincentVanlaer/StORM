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
use lapack::dgbtrf;
use linalg::Matrix;
use model::StellarModel;
use num::Float;
use std::{fmt::Display, mem::transmute};

use crate::{
    jacobian::{Interpolator, NonRotating1D},
    stepper::{MagnusGL2, MagnusGL4, MagnusGL6, Step},
};

mod jacobian;
mod linalg;
mod model;
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

    const KL: usize = 5;
    const KU: usize = 2;
    const STEPS: usize = 6199;
    const ALEN: usize = 4 * STEPS + 4;
    const FREQS: usize = 1000;

    let mut ipiv: [i32; ALEN] = [0; ALEN];
    let mut info: i32 = 0;
    let mut dets = [0.0; FREQS];

    for l in 0..FREQS {
        let mut storage = [[0.0; 2 * KL + KU + 1]; ALEN];
        let freq = 1.62 + (1. / FREQS as f64) * l as f64;
        for i in 0..STEPS {
            // let pos1 = (i as f64) / (STEPS as f64);
            // let pos2 = ((i + 1) as f64) / (STEPS as f64);
            let pos1 = model.r_coord[i] / model.radius;
            let pos2 = model.r_coord[i + 1] / model.radius;
            let matrix = MagnusGL6::step(&interpolator, pos1, pos2, freq);

            for j in 0..4 {
                for k in 0..4 {
                    storage[i * 4 + j][KL + KU + k - j + 2] = matrix[j][k];
                }
                storage[(i + 1) * 4 + j][KL + KU - 2] = -1.0;
            }
        }

        let lower_boundary = interpolator.lower_boundary(freq);
        let upper_boundary = interpolator.upper_boundary(freq);

        for j in 0..4 {
            for k in 0..2 {
                storage[j][KL + KU + k - j] = lower_boundary[j * 2 + k];
                storage[ALEN - 4 + j][KL + KU + k - j + 2] = upper_boundary[j * 2 + k];
            }
        }

        let storage_ref: &mut [f64; ALEN * (2 * KL + KU + 1)] = unsafe { transmute(&mut storage) };

        unsafe {
            dgbtrf(
                ALEN as i32,
                ALEN as i32,
                KL as i32,
                KU as i32,
                storage_ref,
                (2 * KL + KU + 1) as i32,
                &mut ipiv,
                &mut info,
            )
        };

        if info < 0 {
            panic!("dgbtrf failed")
        }

        let mut det = 1.0;

        for i in 0..ALEN {
            det *= storage[i][KL + KU];
        }

        let mut sgn = 1;

        for i in 0..ALEN {
            if ipiv[i] != (i + 1) as i32 {
                sgn *= -1;
            }
        }

        dets[l] = sgn as f64 * det;
    }

    for (i, det) in dets.iter().enumerate() {
        let freq = 1.62 + (1. / FREQS as f64) * i as f64;
        println!("{freq:.3}: {det}");
    }

    Ok(())
}
