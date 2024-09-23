#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![allow(incomplete_features)]

use color_eyre::Result;
use ndarray::aview0;
use std::io::{self, BufRead};
use storm::dynamic_interface::{get_solvers, DifferenceSchemes};
use storm::helpers::linspace;

use storm::bracket::{Balanced, BracketResult, BracketSearcher as _, Point, SearchBrackets as _};
use storm::model::StellarModel;
use storm::system::adiabatic::{ModelGrid, Rotating1D};

struct Solution {
    bracket: BracketResult,
    eigenvector: Vec<f64>,
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
                    let mut bare_input = StellarModel::from_gsm(args[1])?;
                    if args.len() == 3 {
                        bare_input.overlay_rot(args[2])?;
                    }
                    input = Some(bare_input)
                }
                "scan" => {
                    let ell: u64 = args[1].parse()?;
                    let m: i64 = args[2].parse()?;
                    let lower: f64 = args[3].parse()?;
                    let upper: f64 = args[4].parse()?;
                    let steps: usize = args[5].parse()?;

                    let system = Rotating1D::from_model(input.as_ref().unwrap(), ell, m)?;
                    let searcher = &Balanced { rel_epsilon: 0. };
                    let determinant =
                        get_solvers(&system, DifferenceSchemes::Magnus6, &ModelGrid { scale: 0 });

                    let dets: Vec<_> = linspace(lower, upper, steps)
                        .map(|x| Point {
                            x,
                            f: determinant.det(x),
                        })
                        .collect();

                    solutions.extend(dets.iter().brackets().map(|(point1, point2)| {
                        let res = searcher
                            .search(
                                *point1,
                                *point2,
                                |point| Ok::<_, !>(determinant.det(point)),
                                None,
                            )
                            .into_ok();

                        Solution {
                            eigenvector: determinant.eigenvector(res.freq),
                            bracket: res,
                            ell,
                            m,
                        }
                    }));
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
                            let (chunks, _) = solution.eigenvector.as_chunks::<4>();

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
