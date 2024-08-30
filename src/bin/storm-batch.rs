#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]

use color_eyre::eyre::{eyre, Context};
use color_eyre::Result;
use ndarray::{aview0, s};
use std::io::{self, BufRead};

use storm::bracket::{BracketResult, BracketSearcher as _, Brent, Point};
use storm::model::StellarModel;
use storm::solver::{decompose_system_matrix, DecomposedSystemMatrix};
use storm::stepper::{Colloc2, Magnus2, Magnus6};
use storm::system::adiabatic::{ModelGrid, NonRotating1D};

struct Solution {
    bracket: BracketResult,
    decomposed: DecomposedSystemMatrix,
    ell: u64,
    m: i64,
}

fn main() -> Result<()> {
    let mut input: Option<StellarModel> = None;
    let mut solutions: Vec<Solution> = Vec::new();

    for line in io::stdin().lock().lines() {
        let line = line?;

        for command in line.split(";") {
            let args: Vec<&str> = command.split_whitespace().collect();

            match args[0] {
                "input" => {
                    let mut bare_input = load_model(args[1])?;
                    if args.len() == 3 {
                        bare_input.overlay_rot(&hdf5::File::open(args[2])?)?;
                    }
                    input = Some(bare_input)
                }
                "scan" => {
                    let ell: u64 = args[1].parse()?;
                    let m: i64 = args[2].parse()?;
                    let lower: f64 = args[3].parse()?;
                    let upper: f64 = args[4].parse()?;
                    let steps: usize = args[5].parse()?;

                    let system = NonRotating1D::from_model(input.as_ref().unwrap(), ell, m)?;
                    let system_matrix = |freq: f64| -> Result<DecomposedSystemMatrix> {
                        decompose_system_matrix(&system, &Magnus6 {}, &ModelGrid { scale: 0 }, freq)
                            .or(Err(eyre!("Failed determinant")))
                    };

                    let mut dets = vec![Point { x: 0.0, f: 0.0 }; steps];
                    for i in 0..steps {
                        let freq = lower + i as f64 / (steps - 1) as f64 * (upper - lower);
                        dets[i] = Point {
                            x: freq,
                            f: system_matrix(freq)
                                .wrap_err("Frequency scan failed")?
                                .determinant(),
                        };
                    }

                    solutions.extend(
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
                            .map(|(lower, upper)| {
                                let bracket = (Brent { rel_epsilon: 1e-15 })
                                    .search(
                                        lower,
                                        upper,
                                        |point| system_matrix(point).map(|x| x.determinant()),
                                        None,
                                    )
                                    .expect("Bracket failed");

                                Solution {
                                    decomposed: system_matrix(bracket.freq).unwrap(),
                                    bracket,
                                    ell,
                                    m,
                                }
                            }),
                    );
                }
                "output" => {
                    let output = hdf5::File::create(args[1])?;
                    let include_eigenfunctions: bool = args[2].parse()?;

                    for (i, solution) in solutions.iter().enumerate() {
                        let group = output.create_group(format!("{i:05}").as_str())?;

                        group
                            .new_attr_builder()
                            .with_data(aview0(&solution.bracket.freq))
                            .create("freq")?;

                        group
                            .new_attr_builder()
                            .with_data(aview0(&solution.ell))
                            .create("ell")?;

                        group
                            .new_attr_builder()
                            .with_data(aview0(&solution.m))
                            .create("m")?;

                        if include_eigenfunctions {
                            let eigenvector = solution.decomposed.eigenvector();

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
                    }
                    solutions.clear();
                }
                arg => {
                    panic!("Unknown command '{arg}'")
                }
            }
        }
    }

    Ok(())
}

fn load_model(input: &str) -> Result<StellarModel> {
    StellarModel::from_gsm(&hdf5::File::open(input)?)
}
