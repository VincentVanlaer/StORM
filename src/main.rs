#![feature(slice_as_chunks)]
#![feature(generic_const_exprs)]

use std::time::Instant;

use clap::Parser;
use color_eyre::eyre::{eyre, Context};
use color_eyre::Result;
use ndarray::{aview0, s};

use storm::bracket::{Balanced, BracketSearcher as _, Point};
use storm::model::StellarModel;
use storm::solver::{decompose_system_matrix, DecomposedSystemMatrix};
use storm::stepper::{Colloc2, Magnus2, Magnus6};
use storm::system::adiabatic::{ModelGrid, NonRotating1D};

#[derive(Parser)]
#[command()]
struct Main {
    #[arg(long)]
    input: String,
    #[arg(long)]
    overlay_rot: Option<String>,
    #[arg(long)]
    output: String,
    #[arg(long)]
    lower: f64,
    #[arg(long)]
    upper: f64,
    #[arg(long)]
    n_steps: usize,
    #[arg(long)]
    degree: u64,
    #[arg(long)]
    order: i64,
    #[arg(long)]
    eigenfunctions: bool,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Main::parse();
    let mut model = StellarModel::from_gsm(&hdf5::File::open(&args.input)?)?;
    let output = hdf5::File::create(args.output)?;

    if let Some(overlay) = args.overlay_rot {
        model.overlay_rot(&hdf5::File::open(&overlay)?)?;
    }

    output
        .new_dataset_builder()
        .with_data(&(&model.r_coord / model.radius).slice(s![1..]))
        .create("x")?;

    let system = NonRotating1D::from_model(&model, args.degree, args.order)?;
    let mut dets = vec![Point { x: 0.0, f: 0.0 }; args.n_steps];

    let start = Instant::now();

    let system_matrix = |freq: f64| -> Result<DecomposedSystemMatrix> {
        decompose_system_matrix(&system, &Magnus6 {}, &ModelGrid { scale: 0 }, freq)
            .or(Err(eyre!("Failed determinant")))
    };

    for i in 0..args.n_steps {
        let freq = args.lower + i as f64 / (args.n_steps - 1) as f64 * (args.upper - args.lower);
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
        .map(|(lower, upper)| {
            (Balanced { rel_epsilon: 1e-12 }).search(
                lower,
                upper,
                |point| system_matrix(point).map(|x| x.determinant()),
                None,
            )
        })
        .collect();
    for (i, solution) in solutions.iter().enumerate() {
        match solution {
            Ok(result) => {
                println!("{:.20} {}", result.freq, result.evals);

                let group = output.create_group(format!("{i:05}").as_str())?;

                group
                    .new_attr_builder()
                    .with_data(aview0(&result.freq))
                    .create("freq")?;

                if args.eigenfunctions {
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
            Err(_) => println!("Failed to bracket root"),
        }
    }

    Ok(())
}
