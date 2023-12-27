#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use color_eyre::Result;
use hdf5::File as hdf5File;
use linalg::Matrix;
use model::StellarModel;
use num::Float;
use std::{fmt::Display, fs::File, io::Write, f64::consts::PI};

use crate::{
    jacobian::{NonRotating1D, StretchedString, constant_speed, parabola, linear_piecewise, smoothened_linear_piecewise},
    solver::{bracket_search, determinant},
    stepper::{MagnusGL6, MagnusGL2, MagnusGL4, MagnusGL8},
};

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
            print!("{:>34.30}", matrix[j][i]);
        }
        println!();
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let f = hdf5File::open("Z0.016_M11.50_logD3.00_aov0.000_fov0.020_mn2186-MS.GSM")?;

    // let model = StellarModel::from_gsm(&f)?;
    // let interpolator = NonRotating1D::from_model(&model, 1)?;
    let interpolator = StretchedString { speed_generator: linear_piecewise };

    const REFINEMENTS: usize = 40;
    const OFFSET: usize = 16;
    const LOWER: f64 = 1.0;
    const UPPER: f64 = 3.0;
    let base: f64 = 2.0.sqrt().sqrt();
    let mut results = [[0.0; 4]; REFINEMENTS];
    let gl8 = MagnusGL8::new();

    for i in 0..REFINEMENTS {
        let steps = base.powi((i + OFFSET) as i32).ceil() as u64;
        let grid = (0..steps + 1)
            .map(|n| 1. / steps as f64 * n as f64)
            .collect();

        println!("{grid:?}");
        results[i][0] = bracket_search(&interpolator, &MagnusGL2 {}, &grid, LOWER, UPPER);
        results[i][1] = bracket_search(&interpolator, &MagnusGL4 {}, &grid, LOWER, UPPER);
        results[i][2] = bracket_search(&interpolator, &MagnusGL6 {}, &grid, LOWER, UPPER);
        results[i][3] = bracket_search(&interpolator, &gl8, &grid, LOWER, UPPER);
    }

    let true_solution = results[REFINEMENTS - 1][3];
    // let true_solution = PI;
    
    let mut f = File::create("a.csv")?;

    println!("True solution: {}", true_solution);

    for i in 0..REFINEMENTS {
        let steps = base.powi((i + OFFSET) as i32).ceil() as u64;
        let root_gl2 = results[i][0] - true_solution;
        let root_gl4 = results[i][1] - true_solution;
        let root_gl6 = results[i][2] - true_solution;
        let root_gl8 = results[i][3] - true_solution;
        write!(f, "{steps:10}, {root_gl2:+.18},  {root_gl4:+.18},  {root_gl6:+.18},  {root_gl8:+.18}\n")?;
    }


    Ok(())
}
