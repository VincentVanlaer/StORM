#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]

use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{eyre, Context};
use color_eyre::Result;
use ndarray::aview0;

use crate::bracket::{BracketResult, BracketSearcher as _, Brent, Point};
use crate::model::StellarModel;
use crate::solver::{decompose_system_matrix, DecomposedSystemMatrix};
use crate::stepper::{Colloc2, Magnus2, Magnus8};
use crate::system::adiabatic::{ModelGrid, NonRotating1D};

mod bracket;
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
    input: String,
    output: String,
    lower: f64,
    upper: f64,
    n_steps: usize,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Main::parse();
    let model = &StellarModel::from_gsm(&hdf5::File::open(&args.input)?)?;
    let output = hdf5::File::create(args.output)?;

    output.new_dataset_builder().with_data(&(&model.r_coord / model.radius)).create("x")?;

    for m in -1..=1 {
        let system = NonRotating1D::from_model(
            model,
            1,
            m,
        )?;
        let mut dets = vec![Point { x: 0.0, f: 0.0 }; args.n_steps];

        let start = Instant::now();

        let system_matrix = |freq: f64| -> Result<DecomposedSystemMatrix> {
            decompose_system_matrix(&system, &Magnus2 {}, &ModelGrid { scale: 0 }, freq)
                .or(Err(eyre!("Failed determinant")))
        };

        for i in 0..args.n_steps {
            let freq = args.lower + i as f64 / args.n_steps as f64 * (args.upper - args.lower);
            dets[i] = Point {
                x: freq,
                f: system_matrix(freq)
                    .wrap_err("Frequency scan failed")?
                    .determinant(),
            };
        }

        println!("Scan done, took {:?}", start.elapsed());

        let solutions: Vec<_> = dets
            .windows(2)
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
            .map(|(lower, upper)| {
                (Brent { rel_epsilon: 1e-15 }).search(lower, upper, |point| {
                    system_matrix(point).map(|x| x.determinant())
                })
            })
            .collect();
        for (i, solution) in solutions.iter().enumerate() {
            match solution {
                Ok(result) => {
                    let eigenvector = system_matrix(result.freq)?.eigenvector();

                    let (chunks, _) = eigenvector.as_chunks::<4>();

                    let mut vec1 = vec![0.0; chunks.len()];
                    let mut vec2 = vec![0.0; chunks.len()];
                    let mut vec3 = vec![0.0; chunks.len()];
                    let mut vec4 = vec![0.0; chunks.len()];

                    for (i, chunk) in chunks.iter().enumerate() {
                        vec1[i] = chunk[0];
                        vec2[i] = chunk[1];
                        vec3[i] = chunk[2];
                        vec4[i] = chunk[3];
                    }

                    let group = output.create_group(format!("{i:05}_{m:02}").as_str())?;

                    group
                        .new_attr_builder()
                        .with_data(aview0(&result.freq))
                        .create("freq")?;

                    group
                        .new_dataset_builder()
                        .with_data(vec1.as_slice())
                        .create("y1")?;
                    group
                        .new_dataset_builder()
                        .with_data(vec2.as_slice())
                        .create("y2")?;
                    group
                        .new_dataset_builder()
                        .with_data(vec3.as_slice())
                        .create("y3")?;
                    group
                        .new_dataset_builder()
                        .with_data(vec4.as_slice())
                        .create("y4")?;
                }
                Err(_) => println!("Failed to bracket root"),
            }
        }
    }

    Ok(())
}
