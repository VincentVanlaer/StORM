#![feature(generic_const_exprs)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]

use std::time::Instant;

use clap::Parser;
use color_eyre::Result;
use model::StellarModel;
use solver::{determinant, Brent};
use stepper::{Colloc2, Magnus2, Magnus8};
use system::adiabatic::{ModelGrid, NonRotating1D};

use crate::solver::{Bisection, BracketResult, BracketSearcher, Point};

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
    n_steps: usize,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Main::parse();
    let model =
        NonRotating1D::from_model(&StellarModel::from_gsm(&hdf5::File::open(args.file)?)?, 1)?;
    let mut dets = vec![Point { x: 0.0, f: 0.0 }; args.n_steps];

    let start = Instant::now();

    for i in 0..args.n_steps {
        let freq = args.lower + i as f64 / args.n_steps as f64 * (args.upper - args.lower);
        let det = determinant(&model, &Magnus8 {}, &ModelGrid { scale: 0 }, freq);
        dets[i] = Point { x: freq, f: det };
    }

    println!("Scan done, took {:?}", start.elapsed());

    dets.windows(2)
        .filter_map(|window| {
            let pair1 = window[0];
            let pair2 = window[1];

            if pair1.f.signum() != pair2.f.signum() {
                Some((pair1, pair2))
            } else {
                None
            }
        })
        .inspect(|(lower, upper)| println!("{lower:?}, {upper:?}"))
        .for_each(|(lower, upper)| {
            let start = Instant::now();

            match (Brent { rel_epsilon: 1e-15 }).search(lower, upper, |x| {
                determinant(&model, &Magnus8 {}, &ModelGrid { scale: 0 }, x)
            }) {
                Ok(BracketResult { freq, evals }) => println!(
                    "Frequency: {freq:0.5}, took {:?} and {evals} evaluations",
                    start.elapsed()
                ),
                Err(_) => println!("Bracket search failed"),
            };
        });

    Ok(())
}
