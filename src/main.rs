#![feature(generic_const_exprs)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]

use color_eyre::Result;

use linalg::Matrix;

use num::Float;
use std::{fmt::Display};



mod linalg;
mod model;
mod solver;
mod stepper;
mod system;

extern crate blas_src;
extern crate lapack_src;

fn print_matrix<T: Float + Display, const ROWS: usize, const COLUMNS: usize>(
    matrix: &Matrix<T, ROWS, COLUMNS>,
) where
    [(); ROWS * COLUMNS]: Sized,
{
    for i in 0..COLUMNS {
        for j in 0..ROWS {
            print!("{:>35.30}", matrix[j][i]);
        }
        println!();
    }
}

fn main() -> Result<()> {
    color_eyre::install()?;

    Ok(())
}
