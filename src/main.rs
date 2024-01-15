#![feature(generic_const_exprs)]
#![feature(iter_map_windows)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]

use clap::Parser;
use color_eyre::Result;
use model::StellarModel;
use solver::determinant;
use stepper::Magnus8;
use system::adiabatic::{ModelGrid, NonRotating1D};

use crate::stepper::{Colloc2, Magnus2};

mod linalg;
mod model;
mod solver;
mod stepper;
mod system;

extern crate blas_src;
extern crate lapack_src;

#[derive(Parser)]
#[command()]
struct Main {
    file: String,
    lower: f64,
    upper: f64,
    n_steps: u64,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Main::parse();
    let model = StellarModel::from_gsm(&hdf5::File::open(args.file)?)?;

    for i in 0..args.n_steps {
        let freq = args.lower + i as f64 / args.n_steps as f64 * (args.upper - args.lower); 
        let det = determinant(
            &NonRotating1D::from_model(&model, 1)?,
            &Colloc2 {},
            &ModelGrid { scale: 0 },
            freq,
        );

        println!("{}: {}", freq, det);
    }

    Ok(())
}
